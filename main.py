import os
import requests
import zipfile
import gdown
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import List
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Setup FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BIOPORTAL_API_KEY = os.getenv("BIOPORTAL_API_KEY")

# Download and extract model
def download_and_extract_model():
    zip_path = "./model/BioBert3.zip"
    folder_path = "./model/BioBert3"

    if not os.path.exists(folder_path):
        os.makedirs("./model", exist_ok=True)
        file_id = "1kRZ3H8BiEMUvEajoPfC2pICQWYGdQoD9"
        url = f"https://drive.google.com/uc?id={file_id}"
        print("Downloading model...")
        gdown.download(url, zip_path, quiet=False)
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall("./model")
        print("Model ready.")
    else:
        print("Model already exists.")

download_and_extract_model()

# Load model
model_path = "./model/BioBert3"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# SQLite setup
DATABASE_URL = "sqlite:///./diagnosis.db"
Base = declarative_base()

class DiagnosisRecord(Base):
    __tablename__ = "diagnoses"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(100), index=True)
    text = Column(Text)
    diagnosis = Column(String(100))
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(bind=engine)

# Labels
disease_labels = [
    "(vertigo) Paroymsal  Positional Vertigo", "AIDS", "Acne", "Alcoholic hepatitis",
    "Allergy", "Arthritis", "Bronchial Asthma", "Cervical spondylosis", "Chicken pox",
    "Chronic cholestasis", "Common Cold", "Dengue", "Diabetes ", "Dimorphic hemmorhoids(piles)",
    "Drug Reaction", "Fungal infection", "GERD", "Gastroenteritis", "Heart attack",
    "Hepatitis B", "Hepatitis C", "Hepatitis D", "Hepatitis E", "Hypertension ",
    "Hyperthyroidism", "Hypoglycemia", "Hypothyroidism", "Impetigo", "Jaundice",
    "Malaria", "Migraine", "Osteoarthristis", "Paralysis (brain hemorrhage)",
    "Peptic ulcer diseae", "Pneumonia", "Psoriasis", "Tuberculosis", "Typhoid",
    "Urinary tract infection", "Varicose veins", "hepatitis A"
]

class PatientInput(BaseModel):
    user_id: str
    text: str

def query_bioportal(query, api_key):
    base_url = "http://data.bioontology.org/search"
    headers = {"Authorization": f"apikey token={api_key}"}
    params = {"q": query, "ontologies": "SNOMEDCT", "pagesize": 1}
    resp = requests.get(base_url, headers=headers, params=params)
    if resp.status_code == 200:
        results = resp.json().get("collection", [])
        if results:
            return results[0].get("definition") or "No description available."
    return "No description available."

def review_diagnosis_with_gpt(text, diagnosis, description):
    prompt = (
        f"Symptoms: {text}\n"
        f"Initial diagnosis: {diagnosis}\n"
        f"Description: {description}\n"
        f"Please respond in the following strict format:\n"
        f"Diagnosis: <only the final diagnosis as a medical term, no explanation>\n"
        f"Description: <a one-line medical explanation of the diagnosis>\n"
        f"Respond only in English and don't add anything else."
    )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
    if response.status_code == 200:
        reply = response.json()['choices'][0]['message']['content'].strip()
        diagnosis_final = diagnosis
        description_final = description
        for line in reply.split("\n"):
            if "diagnosis" in line.lower():
                diagnosis_final = line.split(":", 1)[-1].strip()
            elif "description" in line.lower():
                description_final = line.split(":", 1)[-1].strip()
        return diagnosis_final, description_final
    return diagnosis, description

@app.post("/diagnose/")
def diagnose(input_data: PatientInput):
    try:
        inputs = tokenizer(input_data.text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            top_prob, top_class = torch.max(probs, dim=1)
        diagnosis = disease_labels[top_class.item()]
        description = query_bioportal(diagnosis, BIOPORTAL_API_KEY)
        diagnosis_final, description_final = review_diagnosis_with_gpt(input_data.text, diagnosis, description)
        if description_final.lower() == "no description available.":
            description_final = query_bioportal(diagnosis_final, BIOPORTAL_API_KEY)
        db = SessionLocal()
        record = DiagnosisRecord(
            user_id=input_data.user_id,
            text=input_data.text,
            diagnosis=diagnosis_final,
            description=description_final
        )
        db.add(record)
        db.commit()
        db.close()
        return {"diagnosis": diagnosis_final, "description": description_final}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class DiagnosisOut(BaseModel):
    id: int
    user_id: str
    text: str
    diagnosis: str
    description: str
    created_at: datetime
    class Config:
        orm_mode = True

@app.get("/history/", response_model=List[DiagnosisOut])
def get_history(user_id: str):
    try:
        db = SessionLocal()
        records = db.query(DiagnosisRecord).filter(DiagnosisRecord.user_id == user_id).order_by(DiagnosisRecord.created_at.desc()).all()
        db.close()
        return records
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

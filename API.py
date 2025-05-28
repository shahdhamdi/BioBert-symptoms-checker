import os
import requests
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

# Load environment variables
load_dotenv()

# Setup FastAPI app
app = FastAPI()
app.add_middleware(
    CROSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load API keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BIOPORTAL_API_KEY = os.getenv("BIOPORTAL_API_KEY")

# Load model
model_path = r"D:\downloads\BioBert3"
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
    params = {
        "q": query,
        "ontologies": "SNOMEDCT",
        "pagesize": 1
    }
    resp = requests.get(base_url, headers=headers, params=params)
    if resp.status_code == 200:
        results = resp.json().get("collection", [])
        if results:
            entry = results[0]
            description = entry.get("definition") or "No description available."
            return description
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
        result = response.json()
        reply = result['choices'][0]['message']['content'].strip()

        diagnosis_confirmed = diagnosis
        description_confirmed = description

        for line in reply.split("\n"):
            if "diagnosis" in line.lower():
                diagnosis_confirmed = line.split(":", 1)[-1].strip()
            elif "description" in line.lower():
                description_confirmed = line.split(":", 1)[-1].strip()

        return diagnosis_confirmed, description_confirmed
    else:
        return diagnosis, description

@app.post("/diagnose/")
def diagnose(input_data: PatientInput):
    try:
        text = input_data.text
        user_id = input_data.user_id

        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            top_prob, top_class = torch.max(probs, dim=1)

        diagnosis = disease_labels[top_class.item()]
        description = query_bioportal(diagnosis, BIOPORTAL_API_KEY)
        diagnosis_final, description_final = review_diagnosis_with_gpt(text, diagnosis, description)
        description_final_bioportal = query_bioportal(diagnosis_final, BIOPORTAL_API_KEY)

        final_description = description_final if description_final.lower() != "no description available." else description_final_bioportal

        db = SessionLocal()
        record = DiagnosisRecord(
            user_id=user_id,
            text=text,
            diagnosis=diagnosis_final,
            description=final_description
        )
        db.add(record)
        db.commit()
        db.close()

        return {
            "diagnosis": diagnosis_final,
            "description": final_description
        }

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

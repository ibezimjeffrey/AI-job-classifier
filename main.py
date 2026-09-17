import re
from fastapi import FastAPI
from pydantic import BaseModel
from profanity_check import predict_prob

app = FastAPI(title="STEP Moderation Microservice")

class JobPostRequest(BaseModel):
    title: str
    description: str

# -----------------------------------------------------------------------------
# HARD BLOCKLIST PHRASES & REGEX
# -----------------------------------------------------------------------------
EXPLICIT_PHRASES = [
    # Academic Fraud
    "help me cheat", "write my exam", "do my exam", "take my exam", "write my test",
    "do my test", "solve my assignment for me", "impersonate me", "change my grade",
    "upgrade cgpa", "cgpa boost", "portal hack", "hack portal", "leak exam", "exam runz",
    "c12", "c-12", "exam expo", "exam runs", "do my cbt",
    
    # Explicit Adult & Dating
    "eat me out", "suck my", "choke me", "lick me", "spank me", "f*ck", "fuck",
    "hookup", "escort service", "sugar daddy", "sugar mummy", "sugar mama",
    "send nudes", "sell nudes", "nude pics", "sex for money", "paid sex", "sex worker",
    
    # Violence & Crime
    "help me beat", "help me kill", "help me stab", "help me shoot", "help me rape",
    
    # Scams & Off-Platform
    "double your money", "triple your money", "investment platform guaranteed",
    "ponzi", "cash flip"
]

HARD_BLOCKLIST_PATTERNS = [
    r"\b(beat|kill|stab|sh[o0]{2}t|r[a@]pe|assault|murder|kidnap)\b",
    r"\b(marijuana|cannabis|cocaine|heroin|codeine|refnol|tramadol)\b",
    r"\b(steal|rob|burgle|extort|blackmail)\b",
    r"\b(portal hack|hack portal|upgrade cgpa|cgpa boost)\b",
    r"\b(double your money|investment platform|ponzi)\b"
]

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/classify-job")
def classify_job(job: JobPostRequest):
    full_text = f"{job.title} {job.description}".strip().lower()

    # 1. Structural Checks (Phone numbers / URLs)
    url_pattern = r'https?://\S+|www\.\S+|\S+\.(com|ng|net|org|io|xyz)\S*'
    if re.search(url_pattern, full_text):
        return {"status": "REJECTED", "category": "spam", "confidence": 1.0, "reason": "Contains external URL"}

    if re.search(r'\b0[789][01]\d{8}\b', full_text):
        return {"status": "FLAG_FOR_REVIEW", "category": "suspicious", "confidence": 0.8, "reason": "Contains phone number"}

    # 2. Hard Blocklist Regex
    for pattern in HARD_BLOCKLIST_PATTERNS:
        if re.search(pattern, full_text):
            return {"status": "REJECTED", "category": "community_violation", "confidence": 1.0, "reason": "Safety policy violation"}

    # 3. Explicit Phrase Check
    for phrase in EXPLICIT_PHRASES:
        pattern = r'\b' + re.escape(phrase) + r'\b'
        if re.search(pattern, full_text):
            return {"status": "REJECTED", "category": "community_violation", "confidence": 1.0, "reason": f"Explicit phrase matched: '{phrase}'"}

    # 4. Profanity Check
    toxicity_score = float(predict_prob([full_text])[0])
    if toxicity_score > 0.65:
        return {"status": "REJECTED", "category": "inappropriate_content", "confidence": round(toxicity_score, 4), "reason": "Inappropriate content"}

    # Default: If no policy violation was hit, APPROVE IT IMMEDIATELY
    return {
        "status": "APPROVED",
        "category": "job_okay",
        "confidence": 1.0,
        "reason": "Passed safety checks"
    }
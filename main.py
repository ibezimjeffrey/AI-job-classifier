import os
import re
import numpy as np
import joblib
import onnxruntime as ort
from tokenizers import Tokenizer
from fastapi import FastAPI
from pydantic import BaseModel
from profanity_check import predict_prob
from huggingface_hub import hf_hub_download

app = FastAPI(title="STEP Moderation Microservice")

# --- DOWNLOAD & LOAD PUBLIC ONNX MODEL VIA XENOVA REPO ---
REPO_ID = "Xenova/all-MiniLM-L6-v2"
print("Downloading/Loading ONNX model from Xenova repo...")
MODEL_PATH = hf_hub_download(repo_id=REPO_ID, filename="onnx/model_quantized.onnx")
TOKENIZER_PATH = hf_hub_download(repo_id=REPO_ID, filename="tokenizer.json")

session = ort.InferenceSession(MODEL_PATH)
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
classifier = joblib.load("step_job_moderator.joblib")

# =============================================================================
# EXPLICIT PHRASE LIST
# Checked with word boundaries to prevent false positives (e.g. "sex" in "hex")
# =============================================================================
EXPLICIT_PHRASES = [
    # Academic Malpractice
    "help me cheat", "write my exam", "do my exam", "take my exam",
    "write my test", "do my test", "solve my assignment for me",
    "impersonate me", "change my grade", "upgrade cgpa", "cgpa boost",
    "portal hack", "hack portal", "leak exam", "exam runz", "c12", "c-12",
    "exam expo", "exam runs", "do my cbt",

    # Sexual
    "eat me out", "suck my", "choke me", "lick me", "lick my",
    "spank me", "f*ck", "hookup", "escort service",
    "sugar daddy", "sugar mummy", "sugar mama", "send nudes",
    "sell nudes", "nude pics", "sex for money", "paid sex",
    "date me tonight",
    "be my companion",
    "looking for a girlfriend",
    "looking for a boyfriend", 
    "spend the night",
    "chill with me tonight",
    "relationship for money",

    # Physical Harm
    "help me beat", "help me kill", "help me stab", "help me shoot",
    "help me rape", "help me assault",

    # Illegal Substances
    "buy weed", "sell weed", "plug for weed", "buy codeine",
    "buy tramadol", "buy refnol", "drug plug",

    # Financial Scams / Off-Platform
    "double your money", "triple your money", "investment platform guaranteed",
    "ponzi", "cash flip", "dm me on whatsapp", "contact on telegram",
    "whatsapp me", "dm on ig", "take this off app",
]

# =============================================================================
# HARD BLOCKLIST — Regex patterns (Layer 1)
# NOTE: Every entry MUST end with a comma. Missing commas cause silent
# string concatenation which breaks the patterns below.
# =============================================================================
HARD_BLOCKLIST = [
    # ---- Physical Violence, Harm & Weapons ----
    r"\b(beat|kill|stab|sh[o0]{2}t|r[a@]pe|assault|molest|harass|abuse|murder|strangle|kidnap|torture|maim|poison)\b",
    r"\b(gun|pistol|kn[i1]fe|dagger|blade|machete|bomb|explosive|ammunition|bullet|rifle)\b",
    r"\b(su[i1]c[i1]de|self[- ]?harm|cut my wrist|slit my|hang myself)\b",

    # ---- Dating / Companion Requests ----
    r"\b(date me|be my date|looking for a girlfriend|looking for a boyfriend|romantic|companionship|cuddle|spend time with me|chill with me tonight|relationship for money)\b",
    r"\b(sugar daddy|sugar mummy|sugar mama|hookup|escort|paid dating|rent a girlfriend|rent a boyfriend)\b",
    # ---- Sexual Explicit Content ----
    r"\b(f+u+c+k+|fck|b[a@]ng me|finger me|sex work|camshow|stripper|lapdance)\b",
    r"\b(hookup|escort|sugar daddy|sugar mummy|sugar mama|pimp|pornography)\b",
    r"\b(nude[s]?|nude pics|send nudes|sell nudes|n[u0]d[e3])\b",
    r"\b(boobs|penis|vagina|cunt|dick|cock|pussy|anal|clit|tits|cum)\b",
    r"\b(sexual assault|sex toy|sex for money|sex tonight)\b",

    # ---- Illegal Drugs & Narcotics ----
    r"\b(marijuana|cannabis|cocaine|heroin|meth|crack|codeine|refnol|tramadol|skunk|ecstasy|mdma|xanax|lsd|shrooms|psychedelic)\b",
    r"\b(buy weed|sell weed|plug for|drug dealer|drug plug|loud plug)\b",

    # ---- Crime & Theft ----
    r"\b(steal|rob|burgle|snatch|pickpocket|carjack|extort|blackmail|kidnap|ransom|shoplift)\b",
    r"\b(help me steal|help me rob|break into|bypass lock|stolen phone|stolen laptop|icloud unlock)\b",

    # ---- Academic Malpractice & Portal Fraud ----
    r"\b(write my exam|do my exam|take my exam|impersonate me|exam impersonation)\b",
    r"\b(change my grade|portal hack|hack portal|leak exam|exam runz|exam expo|c12|exam runs)\b",
    r"\b(hack school|upgrade cgpa|cgpa boost|forge transcript|fake degree|fake certificate|secret answer)\b",
    r"\b(write my thesis for me|do my assignment for me|write my project for me)\b",

    # ---- Financial Scams, Ponzi & Off-Platform ----
    r"\b(double your money|triple your money|investment platform|ponzi|crypto drop|cash flip|phishing)\b",
    r"\b(carding|cc fullz|work from home 50k daily|pay fee first|upfront payment|bvn clone|nin bypass)\b",

    # ---- Off-Platform Redirect Signals ----
    r"\b(whatsapp me|dm me on whatsapp|contact on telegram|dm on ig|take this off app|call me on)\b",
    r"\b(0[789][01]\d{8})\b",  # Nigerian phone number pattern in post body

    # ---- Cultism & Hate Speech ----
    r"\b(cult|cultism|fraternity initiation|axemen|baggar|black axe|vikings|pirates|eiye|confraternity)\b",
    r"\b(nigger|nigga|faggot|kike|chink|spic|tranny)\b",
]

# =============================================================================
# SEMANTIC ANCHORS — Used in Layer 3 for contextual relevance check
# Using 5 diverse anchors and taking the MAX similarity is more robust
# than a single anchor.
# =============================================================================
VALID_JOB_ANCHORS = [
    "I need someone to help me with a task on campus",
    "Looking for a student to provide a service for payment",
    "Need a skilled student to complete a job or errand on campus",
    "Hiring a fellow student to help with work design or tutoring",
    "Campus job request for design writing tutoring photography or errand running",
]


class JobPostRequest(BaseModel):
    title: str
    description: str


# =============================================================================
# HELPERS
# =============================================================================

def get_onnx_embedding(text: str) -> np.ndarray:
    """Generates a semantic vector embedding using the quantized ONNX model."""
    encoded = tokenizer.encode(text)
    input_ids = np.array([encoded.ids], dtype=np.int64)
    attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
    token_type_ids = np.zeros_like(input_ids, dtype=np.int64)
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }
    outputs = session.run(None, inputs)
    return np.mean(outputs[0][0], axis=0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def title_description_mismatch(title: str, description: str) -> bool:
    stop_words = {
        "i", "a", "an", "the", "need", "want", "looking", "for",
        "to", "my", "me", "someone", "please", "urgent", "help",
        "campus", "student", "budget", "cheap", "affordable", "good",
        "rate", "asap", "quickly", "available", "needed", "dm", "contact"
    }

    title_keywords = set(title.lower().split()) - stop_words
    desc_keywords = set(description.lower().split()) - stop_words

    # Only flag if title has meaningful words and NONE overlap with description
    if len(title_keywords) >= 2 and len(title_keywords & desc_keywords) == 0:
        return True

    return False

def is_contextually_inappropriate(text: str) -> bool:
    """
    Compares semantic embedding of the post against 5 valid job anchors.
    Takes the MAX similarity — if the post is close to ANY valid anchor
    it passes. Threshold raised to 0.25 from original 0.20.
    """
    vec_text = get_onnx_embedding(text)
    similarities = [
        cosine_similarity(vec_text, get_onnx_embedding(anchor))
        for anchor in VALID_JOB_ANCHORS
    ]
    return max(similarities) < 0.25


def structural_checks(text: str):
    """
    Layer 0 — Fast structural/format checks before any ML inference.
    Returns (status, reason) or (None, None) if clean.
    """
    # Minimum word count
    if len(text.split()) < 3:
        return "REJECTED", "Too few words to be a valid job post (Layer 0)"

    # Maximum length guard — prevents prompt injection via huge posts
    if len(text) > 2000:
        return "REJECTED", "Post exceeds maximum allowed length (Layer 0)"

    # External URL detection — no legitimate campus job needs an external link
    url_pattern = r'https?://\S+|www\.\S+|\S+\.(com|ng|net|org|io|xyz|click)\S*'
    if re.search(url_pattern, text, re.IGNORECASE):
        return "REJECTED", "Contains external URL (Layer 0)"

    # Repetitive characters — classic spam signal (AAAAA, !!!!!, .......)
    if re.search(r'(.)\1{5,}', text):
        return "REJECTED", "Repetitive character spam detected (Layer 0)"

    # Excessive exclamation marks
    if text.count('!') > 4:
        return "FLAG_FOR_REVIEW", "Excessive exclamation marks (Layer 0)"

    # All-caps detection (>70% uppercase letters, min 10 alpha chars)
    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 10 and (sum(1 for c in letters if c.isupper()) / len(letters)) > 0.70:
        return "FLAG_FOR_REVIEW", "Excessive capitalisation — possible spam (Layer 0)"

    # Nigerian phone number embedded in post body
    if re.search(r'\b0[789][01]\d{8}\b', text):
        return "FLAG_FOR_REVIEW", "Contains phone number — possible off-platform redirect (Layer 0)"

    # Social platform redirect keywords
    redirect_signals = ["whatsapp", "telegram", "dm me", "dm on ig", "call me", "contact me privately"]
    for signal in redirect_signals:
        if signal in text:
            return "FLAG_FOR_REVIEW", f"Off-platform redirect signal detected: '{signal}' (Layer 0)"

    return None, None


def match_explicit_phrase(text: str):
    """
    Word-boundary-safe phrase matching to prevent false positives
    (e.g. 'sex' matching 'hexagonal').
    """
    for phrase in EXPLICIT_PHRASES:
        pattern = r'\b' + re.escape(phrase) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            return phrase
    return None


# =============================================================================
# MAIN ENDPOINT
# =============================================================================
@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/classify-job")
def classify_job(job: JobPostRequest):
    full_text = f"{job.title} {job.description}".strip()
    clean_text = full_text.lower()

    # ------------------------------------------------------------------
    # LAYER 0 — Structural / Format Checks
    # ------------------------------------------------------------------
    status, reason = structural_checks(clean_text)
    if status == "REJECTED":
        return {"status": "REJECTED", "category": "spam", "confidence": 1.0, "reason": reason}
    if status == "FLAG_FOR_REVIEW":
        return {"status": "FLAG_FOR_REVIEW", "category": "suspicious", "confidence": 0.75, "reason": reason}

    # ------------------------------------------------------------------
    # LAYER 1A — Hard Blocklist Regex
    # ------------------------------------------------------------------
    for pattern in HARD_BLOCKLIST:
        if re.search(pattern, clean_text, re.IGNORECASE):
            return {
                "status": "REJECTED",
                "category": "community_violation",
                "confidence": 1.0,
                "reason": "Safety policy violation — hard blocklist (Layer 1A)",
            }

    # ------------------------------------------------------------------
    # LAYER 1B — Explicit Phrase Matching (word-boundary safe)
    # ------------------------------------------------------------------
    triggered_phrase = match_explicit_phrase(clean_text)
    if triggered_phrase:
        return {
            "status": "REJECTED",
            "category": "community_violation",
            "confidence": 1.0,
            "reason": f"Explicit phrase matched: '{triggered_phrase}' (Layer 1B)",
        }

    # ------------------------------------------------------------------
    # LAYER 2 — Toxicity / Profanity Scoring
    # ------------------------------------------------------------------
    toxicity_score = float(predict_prob([clean_text])[0])
    if toxicity_score > 0.55:
        return {
            "status": "REJECTED",
            "category": "inappropriate_content",
            "confidence": round(toxicity_score, 4),
            "reason": f"High toxicity score: {toxicity_score:.2f} (Layer 2)",
        }
    # Borderline toxicity — flag for human review instead of hard reject
    if toxicity_score > 0.35:
        return {
            "status": "FLAG_FOR_REVIEW",
            "category": "borderline_content",
            "confidence": round(toxicity_score, 4),
            "reason": f"Borderline toxicity score: {toxicity_score:.2f} (Layer 2)",
        }
    if title_description_mismatch(job.title, job.description):
        return {
            "status": "FLAG_FOR_REVIEW",
            "category": "job_unsure",
            "confidence": 0.80,
            "reason": "Title and description appear unrelated (consistency check)",
        }

    # ------------------------------------------------------------------
    # LAYER 3 — Semantic Embedding Relevance Check (ONNX)
    # Multi-anchor: post passes if it is close to ANY valid job anchor
    # ------------------------------------------------------------------
    if is_contextually_inappropriate(clean_text):
        return {
            "status": "REJECTED",
            "category": "spam",
            "confidence": 0.90,
            "reason": "Semantically unrelated to any valid campus job post (Layer 3 ONNX)",
        }

    # ------------------------------------------------------------------
    # LAYER 4 — Custom Campus Scikit-Learn Classifier
    # ------------------------------------------------------------------
    probs = classifier.predict_proba([clean_text])[0]
    predicted_label = classifier.classes_[probs.argmax()]
    confidence = float(probs.max())

    # Hard reject: clearly classified as scam or spam with high confidence
    if predicted_label != "job_okay" and confidence >= 0.65:
        return {
            "status": "REJECTED",
            "category": predicted_label,
            "confidence": round(confidence, 4),
            "reason": f"Classified as '{predicted_label}' with confidence {confidence:.2f} (Layer 4)",
        }

    # Borderline: low confidence across the board — flag for human review
    if confidence < 0.65:
        return {
            "status": "FLAG_FOR_REVIEW",
            "category": "job_unsure",
            "confidence": round(confidence, 4),
            "reason": f"Low classifier confidence: {confidence:.2f} — needs human review (Layer 4)",
        }

    # ------------------------------------------------------------------
    # ALL LAYERS PASSED — APPROVED
    # ------------------------------------------------------------------
    return {
        "status": "APPROVED",
        "category": "job_okay",
        "confidence": round(confidence, 4),
        "reason": "Passed all moderation layers",
    }
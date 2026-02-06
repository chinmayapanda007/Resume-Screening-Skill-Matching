import re
import pdfplumber
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def read_pdf(path: str) -> str:
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + " "
    return clean_text(text)


# Role Extraction
def extract_role(jd_text: str) -> str:
    patterns = [
        r"(job title|job role|role|position)\s*[:\-]\s*(.+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, jd_text, re.IGNORECASE)
        if match:
            role = clean_role_text(match.group(2))
            if role:
                return role

    return extract_role_from_sentence(jd_text)

def extract_role_from_sentence(text: str) -> str:
    sentence_patterns = [
        r"we are looking for (a|an)?\s*(.+?)\s*(who|with|for|to|,|\.)",
        r"we are hiring (a|an)?\s*(.+?)\s*(who|with|for|to|,|\.)",
        r"seeking (a|an)?\s*(.+?)\s*(who|with|for|to|,|\.)"
    ]

    for pattern in sentence_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            role = clean_role_text(match.group(2))
            if role:
                return role

    return "Not Found"

def clean_role_text(role: str) -> str:
    role = role.lower()

    stop_words = ["for", "with", "based", "who", "to", "in", "at"]
    for stop in stop_words:
        if f" {stop} " in role:
            role = role.split(f" {stop} ")[0]

    role = role.strip().title()

    if len(role.split()) > 6 or len(role) < 3:
        return None

    return role


def extract_sections(text: str) -> dict:
    sections = {
        "skills": "",
        "tools": "",
        "experience": "",
        "certifications": ""
    }

    patterns = {
        "skills": r"(skills|qualifications|required skills)(.*?)(experience|responsibilities|$)",
        "tools": r"(tools|technologies|platforms)(.*?)(experience|responsibilities|$)",
        "certifications": r"(certifications|certification)(.*?)(experience|$)",
        "experience": r"(experience)(.*?)(education|$)"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            sections[key] = match.group(2)

    return sections


def extract_phrases(text: str) -> list:
    doc = nlp(text)
    phrases = set()

    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip().lower()
        if 1 <= len(phrase.split()) <= 4:
            phrases.add(phrase)

    return list(phrases)


def extract_experience(text: str) -> dict:
    range_match = re.search(r"(\d+)\s*(to|-)\s*(\d+)\s*years", text)
    if range_match:
        return {"total_experience_years": f"{range_match.group(1)}-{range_match.group(3)}"}

    single_match = re.search(r"(\d+)\+?\s*years", text)
    if single_match:
        return {"total_experience_years": f"{single_match.group(1)}+"}

    return {}


def extract_resume_experience(text: str) -> int:
    match = re.search(r"(\d+)\s+years", text)
    return int(match.group(1)) if match else 0


# JD -> Json
def extract_jd_json(jd_pdf_path: str) -> dict:
    jd_text = read_pdf(jd_pdf_path)
    sections = extract_sections(jd_text)

    return {
        "role": extract_role(jd_text),
        "skills": extract_phrases(sections["skills"]),
        "tools": extract_phrases(sections["tools"]),
        "bonus": extract_phrases(sections["certifications"]),
        "experience_requirements": extract_experience(jd_text)
    }


# Resume → JSON
def extract_resume_json(resume_pdf_path: str) -> dict:
    resume_text = read_pdf(resume_pdf_path)
    sections = extract_sections(resume_text)

    return {
        "skills": extract_phrases(sections["skills"]),
        "tools": extract_phrases(sections["tools"]),
        "certifications": extract_phrases(sections["certifications"]),
        "experience_years": extract_resume_experience(resume_text)
    }


NORMALIZATION_MAP = {
    "api": "api integration",
    "api integrations": "api integration",
    "gen ai": "generative ai",
    "genai": "generative ai",
    "llm": "large language model",
    "llms": "large language model"
}

def normalize_terms(terms: list) -> list:
    normalized = set()
    for term in terms:
        term = term.lower().strip()
        normalized.add(NORMALIZATION_MAP.get(term, term))
    return list(normalized)


def normalize_data(data: dict) -> dict:
    return {
        "skills": normalize_terms(data.get("skills", [])),
        "tools": normalize_terms(data.get("tools", [])),
        "bonus": normalize_terms(data.get("bonus", data.get("certifications", [])))
    }



#Similarity + Overlap
def build_similarity_text(jd: dict, resume: dict):
    jd_text = " ".join(jd["skills"] + jd["tools"] + jd["bonus"])
    resume_text = " ".join(resume["skills"] + resume["tools"] + resume["bonus"])
    return jd_text, resume_text

def cosine_match(jd_text: str, resume_text: str) -> float:
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([jd_text, resume_text])
    return round(cosine_similarity(vectors[0], vectors[1])[0][0] * 100, 2)

def get_overlap(jd: dict, resume: dict) -> dict:
    return {
        "skills": list(set(jd["skills"]) & set(resume["skills"])),
        "tools": list(set(jd["tools"]) & set(resume["tools"])),
        "bonus": list(set(jd["bonus"]) & set(resume["bonus"]))
    }


#FINAL ORCHESTRATOR
def run_pipeline(jd_pdf: str, resume_pdf: str) -> str:
    jd_raw = extract_jd_json(jd_pdf)
    resume_raw = extract_resume_json(resume_pdf)

    jd = normalize_data(jd_raw)
    resume = normalize_data(resume_raw)

    jd_text, resume_text = build_similarity_text(jd, resume)
    score = cosine_match(jd_text, resume_text)
    overlap = get_overlap(jd, resume)

    return (
        f"Role Extracted: {jd_raw['role']}\n\n"
        f"Match Score: {score}%\n\n"
        f"Resume Experience: {resume_raw['experience_years']} years\n\n"
        f"Skill Overlap:\n"
        f"- Skills: {overlap['skills']}\n"
        f"- Tools: {overlap['tools']}\n"
        f"- Bonus: {overlap['bonus']}"
    )


# from pipeline import run_pipeline

if __name__ == "__main__":
    result = run_pipeline(
        jd_pdf=r"C:\Users\chinm\Downloads\Job Description.pdf",
        resume_pdf=r"C:\Users\chinm\Downloads\NidhiChintamani_AI.pdf"
    )
    print(result)

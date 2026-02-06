
import json
import re
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# CONFIG
# -----------------------------
JD_JSON_PATH = "keywords.json"


# -----------------------------
# PDF READER & CLEANER
# -----------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_resume_pdf(file_path: str) -> str:
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + " "
    return clean_text(text)


# -----------------------------
# LOAD JD (FIXED FILE)
# -----------------------------
def load_jd_keywords() -> dict:
    with open(JD_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# RESUME EXTRACTION
# -----------------------------
def extract_resume_keywords(resume_text: str, jd: dict) -> dict:
    extracted = {
        "skills": [],
        "tools": [],
        "bonus": []
    }

    for skill in jd["skills"]:
        if skill.lower() in resume_text:
            extracted["skills"].append(skill)

    for tool in jd["tools"]:
        if tool.lower() in resume_text:
            extracted["tools"].append(tool)

    for bonus in jd["bonus"]:
        if bonus.lower() in resume_text:
            extracted["bonus"].append(bonus)

    return extracted


def extract_experience_years(resume_text: str) -> int:
    match = re.search(r"(\d+)\s+years", resume_text)
    return int(match.group(1)) if match else 0



# -----------------------------
# SIMILARITY HELPERS
# -----------------------------
def build_similarity_text(jd: dict, resume_data: dict):
    jd_text = " ".join(jd["skills"] + jd["tools"] + jd["bonus"])
    resume_text = " ".join(
        resume_data["skills"]
        + resume_data["tools"]
        + resume_data["bonus"]
    )
    return jd_text, resume_text


def calculate_similarity(jd_text: str, resume_text: str) -> float:
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([jd_text, resume_text])
    score = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(score * 100, 2)


def get_overlap(jd: dict, resume_data: dict) -> dict:
    return {
        "skills": list(set(jd["skills"]) & set(resume_data["skills"])),
        "tools": list(set(jd["tools"]) & set(resume_data["tools"])),
        "bonus": list(set(jd["bonus"]) & set(resume_data["bonus"]))
    }


# -----------------------------
# MAIN MATCH FUNCTION
# -----------------------------
def match_resume(resume_pdf_path: str) -> str:
    jd = load_jd_keywords()
    resume_text = read_resume_pdf(resume_pdf_path)

    resume_data = extract_resume_keywords(resume_text, jd)
    experience_years = extract_experience_years(resume_text)

    jd_text, resume_text_for_similarity = build_similarity_text(jd, resume_data)
    similarity_score = calculate_similarity(jd_text, resume_text_for_similarity)
    overlap = get_overlap(jd, resume_data)

    return (
        f"Resume Match Score: {similarity_score}%\n\n"
        f"Extracted Experience: {experience_years} years\n\n"
        f"Skill Overlap:\n"
        f"- Skills: {overlap['skills']}\n"
        f"- Tools: {overlap['tools']}\n"
        f"- Bonus: {overlap['bonus']}"
    )


# -----------------------------
# ENTRY POINT
# -----------------------------
# if __name__ == "__main__":
#     result = match_resume(
#         resume_pdf_path=r"C:\Users\chinm\Downloads\Amol_Avhale_Lead_RPA_Developer.pdf"
#     )
#     print(result)

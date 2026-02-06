Problem Statement

Resume screening is typically:

Manual and time-consuming

Subjective and inconsistent

Difficult to scale for high-volume hiring

Most modern solutions rely heavily on LLMs, which can introduce:

High cost and latency

Data privacy concerns

Limited transparency in decision-making

This project explores how far we can go using classical NLP and solid system design.

Key Features

📄 PDF Parsing – Handles unstructured JD and resume PDFs

🧠 Dynamic Information Extraction

Role

Skills

Tools & technologies

Certifications

Experience

🔁 Term Normalization (e.g., API ↔ API integration)

📊 Resume–JD Matching

TF-IDF vectorization

Cosine similarity scoring

🔍 Explainable Output

Match score

Skill / tool overlap

Extracted experience

⚙️ API Interface

Exposed via FastAPI for easy integration

Tech Stack

Python

FastAPI – API layer

pdfplumber – PDF text extraction

spaCy – NLP & noun-phrase extraction

scikit-learn – TF-IDF & cosine similarity

Regex – Section & experience extraction

Design Philosophy

✅ Deterministic and explainable

🔐 Privacy-friendly (no external API calls)

⚡ Lightweight and fast

🧩 Modular and extensible

Although ChatGPT was used as a development assistant to accelerate coding and iteration, all system design decisions, architecture, and trade-offs were made deliberately with real-world constraints in mind.

Current Status

🚧 Work in progress

Ongoing improvements include:

Reducing noise in extracted skills

Improving normalization accuracy

Enhancing section detection

Refining similarity scoring

Potential Use Cases

Internal HR or ATS tools

Resume shortlisting automation

Candidate–role fit analysis

NLP experimentation and learning projects

Future Enhancements

Multi-resume ranking

Experience-weighted scoring

JD & resume upload via API

Optional GenAI / RAG-based enrichment

UI dashboard

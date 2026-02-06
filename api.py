from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from main import match_resume   

app = FastAPI(title="Resume Screening API")


class ResumeRequest(BaseModel):
    resume_pdf_path: str


@app.post("/match-resume")
def match_resume_api(request: ResumeRequest):
    try:
        result = match_resume(
            resume_pdf_path=request.resume_pdf_path
        )
        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


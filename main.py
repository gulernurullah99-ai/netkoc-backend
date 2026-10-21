import json
from typing import Literal

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

app = FastAPI(title="NetKoc AI API", version="2.0.0")
client = OpenAI()
MODEL_NAME = "gpt-6-luna"


class ImageQuestionRequest(BaseModel):
    imageBase64: str = Field(min_length=20)
    mode: Literal["quick", "teach", "hint"] = "teach"
    gradeLevel: str | None = None


class TextQuestionRequest(BaseModel):
    question: str = Field(min_length=2, max_length=6000)
    mode: Literal["quick", "teach", "hint"] = "teach"
    gradeLevel: str | None = None


class CoachRequest(BaseModel):
    message: str = Field(min_length=2, max_length=6000)
    gradeLevel: str | None = None
    examGoal: str | None = None


def clean_json(text: str) -> dict:
    value = text.strip()
    if value.startswith("```json"):
        value = value[7:]
    elif value.startswith("```"):
        value = value[3:]
    if value.endswith("```"):
        value = value[:-3]
    return json.loads(value.strip())


def level_text(grade_level: str | None) -> str:
    return grade_level.strip() if grade_level else "Ã¶ÄŸrencinin anlayabileceÄŸi uygun seviye"


def mode_instruction(mode: str) -> str:
    if mode == "quick":
        return "KÄ±sa ve doÄŸrudan Ã§Ã¶z; temel iÅŸlem adÄ±mlarÄ±nÄ± gÃ¶ster."
    if mode == "hint":
        return "Nihai cevabÄ± doÄŸrudan verme; ipuÃ§larÄ± ve yÃ¶nlendirici sorular sun."
    return "Soruyu Ã¶ÄŸretici biÃ§imde, adÄ±m adÄ±m ve her adÄ±mÄ±n nedenini aÃ§Ä±klayarak Ã§Ã¶z."


def schema_instruction() -> str:
    return (
        "Sadece geÃ§erli JSON dÃ¶ndÃ¼r. Markdown kod bloÄŸu kullanma. "
        'JSON biÃ§imi: {"subject":"ders","topic":"konu",'
        '"summary":"kÄ±sa Ã¶zet","answer":"adÄ±m adÄ±m aÃ§Ä±klama",'
        '"result":"sonuÃ§","followUp":"devam Ã¶nerisi"}'
    )


def normalize_solution(result: dict) -> dict:
    return {
        "subject": result.get("subject", "Belirlenemedi"),
        "topic": result.get("topic", "Belirlenemedi"),
        "summary": result.get("summary", ""),
        "answer": result.get("answer", "Ã‡Ã¶zÃ¼m Ã¼retilemedi."),
        "result": result.get("result", ""),
        "followUp": result.get("followUp", "Bu adÄ±mÄ± daha basit anlatmamÄ± isteyebilirsin."),
    }


@app.get("/")
def home():
    return {"message": "NetKoc AI backend calisiyor!", "version": "2.0.0"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/solve")
def solve_question(request: ImageQuestionRequest):
    try:
        prompt = (
            "Bu gÃ¶rseldeki Ã¶ÄŸrenci sorusunu dikkatlice incele. Ders ve konuyu belirle. "
            f"Seviye: {level_text(request.gradeLevel)}. "
            f"{mode_instruction(request.mode)} "
            "GÃ¶rsel okunamÄ±yorsa aÃ§Ä±kÃ§a belirt ve tahmin yÃ¼rÃ¼tme. "
            + schema_instruction()
        )
        response = client.responses.create(
            model=MODEL_NAME,
            input=[{"role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": "data:image/jpeg;base64," + request.imageBase64},
            ]}],
        )
        return normalize_solution(clean_json(response.output_text))
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=500, detail="Yapay zeka geÃ§erli JSON dÃ¶ndÃ¼rmedi.") from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/ask-text")
def ask_text(request: TextQuestionRequest):
    try:
        prompt = (
            "AÅŸaÄŸÄ±daki Ã¶ÄŸrenci sorusunun dersini ve konusunu belirle. "
            f"Seviye: {level_text(request.gradeLevel)}. "
            f"{mode_instruction(request.mode)} "
            f"Ã–ÄŸrenci sorusu: {request.question}\n" + schema_instruction()
        )
        response = client.responses.create(model=MODEL_NAME, input=prompt)
        return normalize_solution(clean_json(response.output_text))
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=500, detail="Yapay zeka geÃ§erli JSON dÃ¶ndÃ¼rmedi.") from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/coach")
def coach(request: CoachRequest):
    try:
        prompt = (
            "Sen NetKoÃ§ adlÄ± destekleyici ve gerÃ§ekÃ§i bir eÄŸitim koÃ§usun. "
            "Somut, uygulanabilir ve Ã¶lÃ§Ã¼lebilir Ã¶neriler ver. "
            f"SÄ±nÄ±f seviyesi: {level_text(request.gradeLevel)}. "
            f"SÄ±nav veya hedef: {request.examGoal or 'belirtilmedi'}. "
            f"Ã–ÄŸrencinin mesajÄ±: {request.message}\n"
            "Sadece geÃ§erli JSON dÃ¶ndÃ¼r. Markdown kullanma. "
            'JSON biÃ§imi: {"title":"baÅŸlÄ±k","advice":"aÃ§Ä±klama",'
            '"steps":["1. adÄ±m","2. adÄ±m","3. adÄ±m"],'
            '"todayTask":"bugÃ¼nkÃ¼ gÃ¶rev"}'
        )
        response = client.responses.create(model=MODEL_NAME, input=prompt)
        result = clean_json(response.output_text)
        return {
            "title": result.get("title", "NetKoÃ§ Ã¶nerisi"),
            "advice": result.get("advice", "Ã–neri Ã¼retilemedi."),
            "steps": result.get("steps", []),
            "todayTask": result.get("todayTask", "BugÃ¼n kÃ¼Ã§Ã¼k bir adÄ±mla baÅŸla."),
        }
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=500, detail="Yapay zeka geÃ§erli JSON dÃ¶ndÃ¼rmedi.") from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
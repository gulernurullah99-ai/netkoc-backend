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
    return grade_level.strip() if grade_level else "öğrencinin anlayabileceği uygun seviye"


def mode_instruction(mode: str) -> str:
    if mode == "quick":
        return "Kısa ve doğrudan çöz; yine de temel işlem adımlarını göster."
    if mode == "hint":
        return "Nihai cevabı doğrudan verme. Öğrenciyi düşündüren ipuçları ve yönlendirici sorular sun."
    return "Soruyu öğretici biçimde, adım adım ve her adımın nedenini açıklayarak çöz."


def solution_schema_instruction() -> str:
    return (
        "Sadece geçerli JSON döndür. Markdown kod bloğu kullanma. "
        'JSON biçimi tam olarak şöyle olsun: '
        '{"subject":"ders","topic":"konu","summary":"sorunun kısa özeti",'
        '"answer":"adım adım açıklama","result":"sonuç",'
        '"followUp":"öğrencinin devam sorusu için kısa öneri"}'
    )


@app.get("/")
def home():
    return {"message": "NetKoc AI backend calisiyor!", "version": "2.0.0"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/solve")
def solve_question(request: ImageQuestionRequest):
    try:
        image_url = "data:image/jpeg;base64," + request.imageBase64
        prompt = (
            "Bu görseldeki öğrenci sorusunu dikkatlice incele. "
            "Sorunun ait olduğu dersi ve konuyu belirle. "
            f"Seviye: {level_text(request.gradeLevel)}. "
            f"{mode_instruction(request.mode)} "
            "Görsel okunamıyorsa bunu answer alanında açıkça belirt ve tahmin yürütme. "
            + solution_schema_instruction()
        )
        response = client.responses.create(
            model=MODEL_NAME,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_url},
                    ],
                }
            ],
        )
        result = clean_json(response.output_text)
        return normalize_solution(result)
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=500, detail="Yapay zeka geçerli JSON döndürmedi.") from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/ask-text")
def ask_text(request: TextQuestionRequest):
    try:
        prompt = (
            "Aşağıdaki öğrenci sorusunu incele. Ders ve konuyu belirle. "
            f"Seviye: {level_text(request.gradeLevel)}. "
            f"{mode_instruction(request.mode)} "
            f"Öğrenci sorusu: {request.question}\n"
            + solution_schema_instruction()
        )
        response = client.responses.create(model=MODEL_NAME, input=prompt)
        result = clean_json(response.output_text)
        return normalize_solution(result)
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=500, detail="Yapay zeka geçerli JSON döndürmedi.") from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/coach")
def coach(request: CoachRequest):
    try:
        prompt = (
            "Sen NetKoç adlı destekleyici ve gerçekçi bir eğitim koçusun. "
            "Tanı koyma, garanti verme veya öğrenciyi küçümseme. "
            "Somut, uygulanabilir ve ölçülebilir öneriler ver. "
            f"Sınıf seviyesi: {level_text(request.gradeLevel)}. "
            f"Sınav/hedef: {request.examGoal or 'belirtilmedi'}. "
            f"Öğrencinin mesajı: {request.message}\n"
            "Sadece geçerli JSON döndür. Markdown kod bloğu kullanma. "
            'JSON biçimi tam olarak şöyle olsun: '
            '{"title":"kısa başlık","advice":"açıklama",'
            '"steps":["1. adım","2. adım","3. adım"],'
            '"todayTask":"bugün yapılacak tek net görev"}'
        )
        response = client.responses.create(model=MODEL_NAME, input=prompt)
        result = clean_json(response.output_text)
        return {
            "title": result.get("title", "NetKoç önerisi"),
            "advice": result.get("advice", "Öneri üretilemedi."),
            "steps": result.get("steps", []),
            "todayTask": result.get("todayTask", "Bugün küçük bir adımla başla."),
        }
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=500, detail="Yapay zeka geçerli JSON döndürmedi.") from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


def normalize_solution(result: dict) -> dict:
    return {
        "subject": result.get("subject", "Belirlenemedi"),
        "topic": result.get("topic", "Belirlenemedi"),
        "summary": result.get("summary", ""),
        "answer": result.get("answer", "Çözüm üretilemedi."),
        "result": result.get("result", ""),
        "followUp": result.get("followUp", "Bu adımı daha basit anlatmamı isteyebilirsin."),
    }

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



SAFETY_MESSAGE = (
    "Bu istek NetKoç'un güvenli eğitim kapsamına uygun değil. "
    "NetKoç yalnızca dersler, sınav hazırlığı, çalışma planı ve öğrenci gelişimi konularında yardımcı olabilir."
)

EDUCATION_SCOPE_PROMPT = """
Bir öğrenci uygulaması için kapsam denetimi yap.
Yalnızca şu başlıklar uygundur: okul dersleri, akademik sorular, sınav hazırlığı, ödev desteği,
çalışma planı, öğrenme teknikleri, eğitim motivasyonu ve öğrenci gelişimi.
Cinsel/yetişkin içerik, şiddet, yasa dışı faaliyet, nefret, taciz, kendine zarar verme,
flört/erotik sohbet ve eğitim dışı genel sohbet uygun değildir.
Yanıtın yalnızca ALLOW veya BLOCK olsun.
"""

def moderation_flagged(input_value) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=input_value,
    )
    return bool(result.results[0].flagged)

def education_scope_allowed(text: str) -> bool:
    response = client.responses.create(
        model=MODEL_NAME,
        input=f"{EDUCATION_SCOPE_PROMPT}\nKullanıcı mesajı: {text}",
    )
    return response.output_text.strip().upper().startswith("ALLOW")

def ensure_text_allowed(text: str) -> None:
    if moderation_flagged(text) or not education_scope_allowed(text):
        raise HTTPException(status_code=400, detail=SAFETY_MESSAGE)

def ensure_image_allowed(image_base64: str) -> None:
    data_url = "data:image/jpeg;base64," + image_base64
    moderation_input = [
        {"type": "text", "text": "Öğrenci tarafından yüklenen soru görseli"},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]
    if moderation_flagged(moderation_input):
        raise HTTPException(status_code=400, detail=SAFETY_MESSAGE)
    response = client.responses.create(
        model=MODEL_NAME,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": EDUCATION_SCOPE_PROMPT},
                {"type": "input_image", "image_url": data_url},
            ],
        }],
    )
    if not response.output_text.strip().upper().startswith("ALLOW"):
        raise HTTPException(status_code=400, detail=SAFETY_MESSAGE)

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
        return "Kısa ve doğrudan çöz; temel işlem adımlarını göster."
    if mode == "hint":
        return "Nihai cevabı doğrudan verme; ipuçları ve yönlendirici sorular sun."
    return "Soruyu öğretici biçimde, adım adım ve her adımın nedenini açıklayarak çöz."


def schema_instruction() -> str:
    return (
        "Sadece geçerli JSON döndür. Markdown kod bloğu kullanma. "
        'JSON biçimi: {"subject":"ders","topic":"konu",'
        '"summary":"kısa özet","answer":"adım adım açıklama",'
        '"result":"sonuç","followUp":"devam önerisi"}'
    )


def normalize_solution(result: dict) -> dict:
    return {
        "subject": result.get("subject", "Belirlenemedi"),
        "topic": result.get("topic", "Belirlenemedi"),
        "summary": result.get("summary", ""),
        "answer": result.get("answer", "Çözüm üretilemedi."),
        "result": result.get("result", ""),
        "followUp": result.get("followUp", "Bu adımı daha basit anlatmamı isteyebilirsin."),
    }


@app.get("/")
def home():
    return {"message": "NetKoc AI backend calisiyor!", "version": "2.0.0"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/solve")
def solve_question(request: ImageQuestionRequest):
    ensure_image_allowed(request.imageBase64)
    try:
        prompt = (
             "Yalnızca eğitim amacıyla yanıt ver. Yetişkin/cinsel, tehlikeli, yasa dışı veya eğitim dışı içerik üretme. "
            "Bu görseldeki öğrenci sorusunu dikkatlice incele. Ders ve konuyu belirle. "
            f"Seviye: {level_text(request.gradeLevel)}. "
            f"{mode_instruction(request.mode)} "
            "Görsel okunamıyorsa açıkça belirt ve tahmin yürütme. "
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
        raise HTTPException(status_code=500, detail="Yapay zeka geçerli JSON döndürmedi.") from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/ask-text")
def ask_text(request: TextQuestionRequest):
    ensure_text_allowed(request.question)
    try:
        prompt = (
             "Yalnızca eğitim amacıyla yanıt ver. Yetişkin/cinsel, tehlikeli, yasa dışı veya eğitim dışı içerik üretme. "
            "Aşağıdaki öğrenci sorusunun dersini ve konusunu belirle. "
            f"Seviye: {level_text(request.gradeLevel)}. "
            f"{mode_instruction(request.mode)} "
            f"Öğrenci sorusu: {request.question}\n" + schema_instruction()
        )
        response = client.responses.create(model=MODEL_NAME, input=prompt)
        return normalize_solution(clean_json(response.output_text))
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=500, detail="Yapay zeka geçerli JSON döndürmedi.") from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/coach")
def coach(request: CoachRequest):
    ensure_text_allowed(request.message)
    try:
        prompt = (
             "Yalnızca eğitim amacıyla yanıt ver. Yetişkin/cinsel, tehlikeli, yasa dışı veya eğitim dışı içerik üretme. "
            "Sen NetKoç adlı destekleyici ve gerçekçi bir eğitim koçusun. "
            "Somut, uygulanabilir ve ölçülebilir öneriler ver. "
            f"Sınıf seviyesi: {level_text(request.gradeLevel)}. "
            f"Sınav veya hedef: {request.examGoal or 'belirtilmedi'}. "
            f"Öğrencinin mesajı: {request.message}\n"
            "Sadece geçerli JSON döndür. Markdown kullanma. "
            'JSON biçimi: {"title":"başlık","advice":"açıklama",'
            '"steps":["1. adım","2. adım","3. adım"],'
            '"todayTask":"bugünkü görev"}'
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
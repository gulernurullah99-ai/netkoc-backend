import json
import os
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

import firebase_admin
from fastapi import Depends, FastAPI, Header, HTTPException
from firebase_admin import auth, credentials, firestore
from google.cloud import firestore as google_firestore
from openai import OpenAI
from pydantic import BaseModel, Field

app = FastAPI(title="NetKoc AI API", version="3.0.0")
client = OpenAI()
MODEL_NAME = "gpt-6-luna"
DAILY_FREE_LIMIT = int(os.getenv("DAILY_FREE_LIMIT", "5"))
TURKEY_TZ = ZoneInfo("Europe/Istanbul")


def initialize_firebase_admin() -> None:
    if firebase_admin._apps:
        return
    service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    if service_account_json:
        firebase_admin.initialize_app(
            credentials.Certificate(json.loads(service_account_json))
        )
    else:
        firebase_admin.initialize_app()


initialize_firebase_admin()
db = firestore.client()

SAFETY_MESSAGE = (
    "NetKoç burada öğrenme yolculuğuna destek olmak için var. "
    "Bu soru eğitim kapsamının dışında kalıyor; ancak dersler, sınavlar, "
    "çalışma planları ve motivasyon konularında yardımcı olmaya hazır. 🌟"
)


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


class CoachResponse(BaseModel):
    title: str
    advice: str
    steps: list[str]
    todayTask: str


def current_uid(authorization: str | None = Header(default=None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Oturum doğrulanamadı. Lütfen yeniden giriş yapın.")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        decoded = auth.verify_id_token(token)
        return decoded["uid"]
    except Exception as error:
        raise HTTPException(status_code=401, detail="Oturum süresi doldu. Lütfen yeniden giriş yapın.") from error


def reserve_daily_question(uid: str) -> int:
    day_key = datetime.now(TURKEY_TZ).strftime("%Y-%m-%d")
    reference = db.collection("users").document(uid).collection("usage").document(day_key)
    transaction = db.transaction()

    @google_firestore.transactional
    def update_usage(txn):
        snapshot = reference.get(transaction=txn)
        data = snapshot.to_dict() or {}
        count = int(data.get("count", 0))
        if count >= DAILY_FREE_LIMIT:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Bugünkü {DAILY_FREE_LIMIT} ücretsiz soru hakkı tamamlandı. "
                    "Yarın tekrar kullanılabilir."
                ),
            )
        new_count = count + 1
        txn.set(
            reference,
            {
                "count": new_count,
                "date": day_key,
                "updatedAt": firestore.SERVER_TIMESTAMP,
                "source": "backend",
            },
            merge=True,
        )
        return new_count

    return update_usage(transaction)


def refund_daily_question(uid: str) -> None:
    day_key = datetime.now(TURKEY_TZ).strftime("%Y-%m-%d")
    reference = db.collection("users").document(uid).collection("usage").document(day_key)
    transaction = db.transaction()

    @google_firestore.transactional
    def update_usage(txn):
        snapshot = reference.get(transaction=txn)
        data = snapshot.to_dict() or {}
        count = max(0, int(data.get("count", 0)) - 1)
        txn.set(
            reference,
            {
                "count": count,
                "date": day_key,
                "updatedAt": firestore.SERVER_TIMESTAMP,
                "source": "backend",
            },
            merge=True,
        )

    update_usage(transaction)


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
        '{"subject":"ders","topic":"konu",'
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
        "followUp": result.get("followUp", "Bu adımı daha basit anlatmamı isteyebilirsiniz."),
    }


def moderation_flagged(input_value) -> bool:
    response = client.moderations.create(model="omni-moderation-latest", input=input_value)
    return bool(response.results[0].flagged)


def education_scope_allowed(text: str) -> bool:
    prompt = (
        "Bir öğrenci uygulaması için kapsam denetimi yap. Yalnızca okul dersleri, akademik sorular, "
        "sınav hazırlığı, IELTS, TOEFL, YDS ve diğer dil sınavları, yabancı dil öğrenimi, "
        "ödev desteği, çalışma planı, eğitim motivasyonu ve öğrenci gelişimi uygundur. "
        "Yetişkin/cinsel, şiddet, yasa dışı, nefret, taciz, kendine zarar verme ve eğitim dışı genel sohbet uygun değildir. "
        "Yalnızca ALLOW veya BLOCK yaz.\nKullanıcı mesajı: " + text
    )
    response = client.responses.create(model=MODEL_NAME, input=prompt)
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


@app.get("/")
def home():
    return {"message": "NetKoc AI backend çalışıyor!", "version": "3.0.0"}


@app.get("/health")
def health():
    return {"status": "ok", "dailyLimit": DAILY_FREE_LIMIT}


@app.post("/solve")
def solve_question(request: ImageQuestionRequest, uid: str = Depends(current_uid)):
    ensure_image_allowed(request.imageBase64)
    reserve_daily_question(uid)
    prompt = (
        "Yalnızca eğitim amacıyla yanıt ver. Bu görseldeki öğrenci sorusunu dikkatlice incele. "
        f"Seviye: {level_text(request.gradeLevel)}. {mode_instruction(request.mode)} "
        "Görsel okunamıyorsa açıkça belirt ve tahmin yürütme. " + schema_instruction()
    )
    try:
        response = client.responses.create(
            model=MODEL_NAME,
            input=[{"role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": "data:image/jpeg;base64," + request.imageBase64},
            ]}],
        )
        return normalize_solution(clean_json(response.output_text))
    except json.JSONDecodeError as error:
        refund_daily_question(uid)
        raise HTTPException(status_code=500, detail="Yapay zekâ geçerli yanıt döndürmedi.") from error
    except HTTPException:
        raise
    except Exception as error:
        refund_daily_question(uid)
        raise HTTPException(status_code=500, detail="Soru şu anda yanıtlanamadı.") from error


@app.post("/ask-text")
def ask_text(request: TextQuestionRequest, uid: str = Depends(current_uid)):
    ensure_text_allowed(request.question)
    reserve_daily_question(uid)
    prompt = (
        "Yalnızca eğitim amacıyla yanıt ver. Aşağıdaki öğrenci sorusunun dersini ve konusunu belirle. "
        f"Seviye: {level_text(request.gradeLevel)}. {mode_instruction(request.mode)} "
        f"Öğrenci sorusu: {request.question}\n" + schema_instruction()
    )
    try:
        response = client.responses.create(model=MODEL_NAME, input=prompt)
        return normalize_solution(clean_json(response.output_text))
    except json.JSONDecodeError as error:
        refund_daily_question(uid)
        raise HTTPException(status_code=500, detail="Yapay zekâ geçerli yanıt döndürmedi.") from error
    except HTTPException:
        raise
    except Exception as error:
        refund_daily_question(uid)
        raise HTTPException(status_code=500, detail="Soru şu anda yanıtlanamadı.") from error


@app.post("/coach")
def coach(request: CoachRequest, uid: str = Depends(current_uid)):
    ensure_text_allowed(request.message)
    reserve_daily_question(uid)
    prompt = (
        "Yalnızca eğitim amacıyla yanıt ver. Sen NetKoç adlı destekleyici ve gerçekçi bir eğitim koçusun. "
        "Somut, uygulanabilir ve ölçülebilir öneriler ver. "
        f"Sınıf seviyesi: {level_text(request.gradeLevel)}. "
        f"Sınav veya hedef: {request.examGoal or 'belirtilmedi'}. "
        f"Öğrencinin mesajı: {request.message}\n"
        "Sadece geçerli JSON döndür. Markdown kullanma. "
        '{"title":"başlık","advice":"açıklama",'
        '"steps":["1. adım","2. adım","3. adım"],'
        '"todayTask":"bugünkü görev"}'
    )
    try:
        response = client.responses.parse(
            model=MODEL_NAME,
            input=prompt,
            text_format=CoachResponse,
        )
        result = response.output_parsed
        if result is None:
            raise ValueError("Koç yanıtı ayrıştırılamadı.")
        return result.model_dump()
    except json.JSONDecodeError as error:
        refund_daily_question(uid)
        raise HTTPException(status_code=500, detail="Yapay zekâ geçerli yanıt döndürmedi.") from error
    except HTTPException:
        raise
    except Exception as error:
        refund_daily_question(uid)
        raise HTTPException(status_code=500, detail="Koç yanıtı şu anda alınamadı.") from error

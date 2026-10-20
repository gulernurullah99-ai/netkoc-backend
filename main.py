import json

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel

app = FastAPI()
client = OpenAI()


class QuestionRequest(BaseModel):
    imageBase64: str


@app.get("/")
def home():
    return {"message": "NetKoc AI backend calisiyor!"}


@app.post("/solve")
def solve_question(request: QuestionRequest):
    try:
        image_url = (
            "data:image/jpeg;base64,"
            + request.imageBase64
        )

        response = client.responses.create(
            model="gpt-6-luna",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Bu görseldeki öğrenci sorusunu dikkatlice incele. "
                                "Sorunun ait olduğu dersi ve konuyu belirle. "
                                "Soruyu adım adım, öğrenciye öğretici şekilde çöz. "
                                "Sadece geçerli JSON döndür. "
                                "JSON biçimi tam olarak şöyle olsun: "
                                '{"subject":"ders",'
                                '"topic":"konu",'
                                '"answer":"adım adım çözüm"}'
                            ),
                        },
                        {
                            "type": "input_image",
                            "image_url": image_url,
                        },
                    ],
                }
            ],
        )

        result_text = response.output_text.strip()

        if result_text.startswith("```json"):
            result_text = result_text[7:]

        if result_text.startswith("```"):
            result_text = result_text[3:]

        if result_text.endswith("```"):
            result_text = result_text[:-3]

        result = json.loads(result_text.strip())

        return {
            "subject": result.get("subject", "Belirlenemedi"),
            "topic": result.get("topic", "Belirlenemedi"),
            "answer": result.get("answer", "Çözüm üretilemedi."),
        }

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Yapay zeka geçerli bir JSON cevabı döndürmedi.",
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Alia TTS")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TTSRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Alia TTS"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/tts")
def tts(request: TTSRequest):

    text = request.text.strip()

    if not text:
        return {
            "error": "text kosong"
        }

    return {
        "status": "received",
        "text": text
    }

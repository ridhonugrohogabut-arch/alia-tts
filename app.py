import os
import re
import tempfile
import threading
from pathlib import Path

import gradio as gr
import numpy as np
import scipy.io.wavfile
import spaces

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

import uvicorn

from pocket_tts import TTSModel


# =========================================================
# CONFIG
# =========================================================

MODEL_CONFIG = (
    "hf://anak10thn/pocket-tts-indonesian/"
    "indonesian_6l.yaml"
)

VOICE_PATH = Path(
    os.environ.get(
        "ALIA_REFERENCE_PATH",
        "alia_reference.wav"
    )
)

MAX_TEXT_CHARS = 4500

# Jangan terlalu besar supaya satu proses TTS
# tidak memakan waktu GPU terlalu lama.
MAX_CHUNK_CHARS = 350


# =========================================================
# STARTUP
# =========================================================

print("======================================")
print(" ALIA TTS")
print(" Indonesian Pocket TTS — 6L")
print(" API + Gradio")
print("======================================")

print("Loading Indonesian Pocket TTS...")

model = TTSModel.load_model(
    config=MODEL_CONFIG
)

print("✅ Indonesian Pocket TTS loaded")


# =========================================================
# VOICE
# =========================================================

if not VOICE_PATH.exists():

    raise FileNotFoundError(
        f"""
Voice reference tidak ditemukan:

{VOICE_PATH}

Pastikan file berikut ada di root Space:

alia_reference.wav
"""
    )


print(
    "Loading Alia voice:",
    VOICE_PATH
)

voice_state = (
    model.get_state_for_audio_prompt(
        str(VOICE_PATH),
        truncate=True
    )
)

print("✅ Alia voice loaded")


# Pocket TTS tidak dipanggil paralel.
generation_lock = threading.Lock()


# =========================================================
# TEXT CLEANING
# =========================================================

def normalize_text(text):

    text = str(
        text or ""
    )

    # Normalisasi newline
    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    # Jangan menghapus tanda baca.
    # Tanda baca membantu Pocket TTS
    # menentukan intonasi.
    #
    # Tetapi tanda baca TIDAK akan dibaca
    # sebagai kata.

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# =========================================================
# MARKDOWN CLEANING
# =========================================================

def clean_for_speech(text):

    text = normalize_text(
        text
    )

    # -----------------------------------------------------
    # Code fence
    # -----------------------------------------------------

    text = re.sub(
        r"```[\s\S]*?```",
        lambda m: m.group(0)
        .replace("```", ""),
        text
    )

    # -----------------------------------------------------
    # Bold
    # -----------------------------------------------------

    text = re.sub(
        r"\*\*(.*?)\*\*",
        r"\1",
        text,
        flags=re.DOTALL
    )

    # -----------------------------------------------------
    # Italic
    # -----------------------------------------------------

    text = re.sub(
        r"(?<!\*)\*(.*?)\*(?!\*)",
        r"\1",
        text,
        flags=re.DOTALL
    )

    # -----------------------------------------------------
    # Heading
    # -----------------------------------------------------

    text = re.sub(
        r"^\s*#{1,6}\s+",
        "",
        text,
        flags=re.MULTILINE
    )

    # -----------------------------------------------------
    # Bullet
    # -----------------------------------------------------

    text = re.sub(
        r"^\s*[-*+]\s+",
        "",
        text,
        flags=re.MULTILINE
    )

    # -----------------------------------------------------
    # Numbered list
    # -----------------------------------------------------

    text = re.sub(
        r"^\s*\d+\.\s+",
        "",
        text,
        flags=re.MULTILINE
    )

    # -----------------------------------------------------
    # Markdown links
    # -----------------------------------------------------

    text = re.sub(
        r"([^]+)\][^)]+",
        r"\1",
        text
    )

    # -----------------------------------------------------
    # URL
    # -----------------------------------------------------

    text = re.sub(
        r"https?://\S+",
        "",
        text
    )

    # -----------------------------------------------------
    # Mention
    # -----------------------------------------------------

    text = re.sub(
        r"@([A-Za-z0-9_.-]+)",
        r"\1",
        text
    )

    # -----------------------------------------------------
    # Emoji
    # -----------------------------------------------------

    emoji_pattern = re.compile(
        "["
        "\U0001F300-\U0001FAFF"
        "\U00002700-\U000027BF"
        "\U0001F1E6-\U0001F1FF"
        "]+",
        flags=re.UNICODE
    )

    text = emoji_pattern.sub(
        "",
        text
    )

    # -----------------------------------------------------
    # Whitespace
    # -----------------------------------------------------

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# =========================================================
# SENTENCE SPLITTING
# =========================================================

def split_long_piece(text):

    text = text.strip()

    if not text:

        return []

    if len(text) <= MAX_CHUNK_CHARS:

        return [
            text
        ]

    chunks = []

    while len(text) > MAX_CHUNK_CHARS:

        # Prioritas pertama:
        # koma
        cut = text.rfind(
            ", ",
            0,
            MAX_CHUNK_CHARS
        )

        # Kalau terlalu pendek,
        # cari spasi biasa.
        if cut < int(
            MAX_CHUNK_CHARS * 0.55
        ):

            cut = text.rfind(
                " ",
                0,
                MAX_CHUNK_CHARS
            )

        # Fallback terakhir
        if cut <= 0:

            cut = MAX_CHUNK_CHARS

        chunks.append(
            text[:cut].strip()
        )

        text = text[cut:].strip()

    if text:

        chunks.append(
            text
        )

    return chunks


def split_for_tts(text):

    """
    Memecah berdasarkan akhir kalimat terlebih dahulu.

    . ! ? …
    """

    # Jangan membuang punctuation.
    sentences = re.split(
        r"(?<=[.!?…])\s+",
        text
    )

    chunks = []

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:

            continue

        chunks.extend(
            split_long_piece(
                sentence
            )
        )

    return chunks


# =========================================================
# PAUSE
# =========================================================

def pause_after_chunk(text):

    text = text.rstrip()

    # Ellipsis
    if (
        text.endswith("...")
        or text.endswith("…")
    ):

        return 0.55

    # Excited / question
    if (
        text.endswith("!")
        or text.endswith("?")
    ):

        return 0.42

    # Normal sentence
    if text.endswith("."):

        return 0.32

    # Bukan akhir kalimat
    return 0.10


# =========================================================
# GPU GENERATION
# =========================================================

@spaces.GPU(
    duration=60
)
def generate_chunk(text):

    text = str(
        text or ""
    ).strip()

    if not text:

        return None

    print(
        "🎙️ GPU TTS:"
    )

    print(
        repr(text)
    )

    audio = model.generate_audio(
        voice_state,
        text
    )

    # Tensor → NumPy
    data = (
        audio
        .detach()
        .cpu()
        .numpy()
    )

    data = np.asarray(
        data,
        dtype=np.float32
    )

    data = data.reshape(
        -1
    )

    return data


# =========================================================
# MAIN GENERATOR
# =========================================================

def generate_voice(text):

    text = clean_for_speech(
        text
    )

    if not text:

        raise gr.Error(
            "Teks masih kosong."
        )

    if len(text) > MAX_TEXT_CHARS:

        raise gr.Error(
            "Teks terlalu panjang. "
            f"Maksimal {MAX_TEXT_CHARS} karakter."
        )

    chunks = split_for_tts(
        text
    )

    if not chunks:

        raise gr.Error(
            "Tidak ada teks yang dapat dibacakan."
        )

    print(
        "======================================"
    )

    print(
        "🎙️ ALIA TTS REQUEST"
    )

    print(
        "Characters:",
        len(text)
    )

    print(
        "Chunks:",
        len(chunks)
    )

    print(
        "======================================"
    )

    audio_parts = []

    with generation_lock:

        for index, chunk in enumerate(
            chunks,
            1
        ):

            print(
                f"▶️ Chunk "
                f"{index}/{len(chunks)}"
            )

            audio = generate_chunk(
                chunk
            )

            if audio is None:

                continue

            audio_parts.append(
                audio
            )

            # Jeda natural setelah
            # akhir kalimat.
            pause = pause_after_chunk(
                chunk
            )

            if index < len(chunks):

                silence = np.zeros(
                    int(
                        model.sample_rate
                        * pause
                    ),
                    dtype=np.float32
                )

                audio_parts.append(
                    silence
                )

    if not audio_parts:

        raise gr.Error(
            "Audio gagal dibuat."
        )

    final_audio = np.concatenate(
        audio_parts
    )

    # -----------------------------------------------------
    # Safety normalization
    # -----------------------------------------------------

    peak = np.max(
        np.abs(final_audio)
    )

    if peak > 1.0:

        final_audio = (
            final_audio
            / peak
        )

    # -----------------------------------------------------
    # Temporary output
    # -----------------------------------------------------

    output_path = os.path.join(
        tempfile.gettempdir(),
        "alia.wav"
    )

    scipy.io.wavfile.write(
        output_path,
        model.sample_rate,
        final_audio
    )

    print(
        "🔊 Generated:",
        output_path
    )

    print(
        "======================================"
    )

    return output_path


# =========================================================
# FASTAPI
# =========================================================

api = FastAPI(
    title="Alia TTS API"
)


# GitHub Pages perlu CORS.
api.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],
    allow_credentials=False,
    allow_methods=[
        "*"
    ],
    allow_headers=[
        "*"
    ]
)


# =========================================================
# REQUEST MODEL
# =========================================================

class TTSRequest(
    BaseModel
):

    text: str


# =========================================================
# HEALTH
# =========================================================

@api.get(
    "/health"
)
def health():

    return {
        "status": "ok",
        "service": "Alia TTS",
        "voice": "alia_reference.wav",
        "sample_rate": model.sample_rate
    }


# =========================================================
# TTS API
# =========================================================

@api.post(
    "/tts"
)
def tts(
    request: TTSRequest
):

    text = clean_for_speech(
        request.text
    )

    if not text:

        raise HTTPException(
            status_code=400,
            detail="text kosong"
        )

    if len(text) > MAX_TEXT_CHARS:

        raise HTTPException(
            status_code=413,
            detail=(
                "Teks terlalu panjang. "
                f"Maksimal {MAX_TEXT_CHARS} karakter."
            )
        )

    try:

        output_path = generate_voice(
            text
        )

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="alia.wav",
            background=BackgroundTask(
                safe_remove,
                output_path
            )
        )

    except Exception as error:

        print(
            "❌ API TTS ERROR:"
        )

        print(
            repr(error)
        )

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# =========================================================
# CLEAN TEMP FILE
# =========================================================

def safe_remove(
    path
):

    try:

        if os.path.exists(
            path
        ):

            os.remove(
                path
            )

    except Exception:

        pass


# =========================================================
# GRADIO TEST UI
# =========================================================

with gr.Blocks(
    title="Alia TTS"
) as demo:

    gr.Markdown(
        """
# ❄️ Alia TTS

**Indonesian Pocket TTS — 6 Layer**

Voice reference:
`alia_reference.wav`

API:

`POST /tts`

Health:

`GET /health`
"""
    )

    text_input = gr.Textbox(
        label="Teks",
        placeholder=(
            "Hmph... ketik sesuatu "
            "di sini."
        ),
        lines=8
    )

    generate_button = gr.Button(
        "🔊 Generate Voice",
        variant="primary"
    )

    audio_output = gr.Audio(
        label="Suara Alia",
        type="filepath"
    )

    generate_button.click(
        fn=generate_voice,
        inputs=text_input,
        outputs=audio_output
    )


# =========================================================
# MOUNT GRADIO
# =========================================================

app = gr.mount_gradio_app(
    api,
    demo,
    path="/",
    ssr_mode=False
)


# =========================================================
# LAUNCH
# =========================================================

if __name__ == "__main__":

    print(
        "🚀 Starting Alia TTS..."
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                7860
            )
        )
    )

from flask import Flask, request, send_file, jsonify
import io

app = Flask(__name__)


@app.route("/")
def home():
    return jsonify({
        "status": "online",
        "service": "Alia TTS"
    })


@app.route("/tts", methods=["POST"])
def tts():

    data = request.get_json(silent=True) or {}

    text = data.get("text", "").strip()

    if not text:
        return jsonify({
            "error": "text is required"
        }), 400

    # TTS engine akan kita pasang di langkah berikutnya.
    return jsonify({
        "status": "received",
        "text": text
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=7860
    )

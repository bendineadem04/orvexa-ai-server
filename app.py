import os
import time

from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types

app = Flask(__name__)

CORS(app, origins=[
    "https://bendineadem04.github.io",
    "https://bendineadem04-collab.github.io"
])

# النموذج الأساسي والبديل قابلان للتغيير من إعدادات Render.
PRIMARY_MODEL = os.environ.get(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)

FALLBACK_MODEL = os.environ.get(
    "GEMINI_FALLBACK_MODEL",
    ""
)

SYSTEM_INSTRUCTION = (
    "You are ORVEXA AI, a helpful, intelligent, and honest AI assistant. "
    "Respond in the same language as the user. "
    "Explain things clearly and accurately."
)


@app.get("/")
def home():
    return jsonify({
        "name": "ORVEXA AI Server",
        "status": "online"
    })


@app.get("/health")
def health():
    return jsonify({"ok": True})


def generate_reply(client, model, contents):
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            max_output_tokens=1200
        )
    )

    reply = response.text

    if not reply:
        raise RuntimeError("The AI returned an empty response.")

    return reply


@app.post("/chat")
def chat():
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        return jsonify({
            "error": "API key is not configured."
        }), 500

    data = request.get_json(silent=True) or {}
    messages = data.get("messages", [])

    if not isinstance(messages, list) or not messages:
        return jsonify({
            "error": "No messages provided."
        }), 400

    contents = []

    for message in messages[-20:]:
        if not isinstance(message, dict):
            continue

        role = message.get("role")
        text = message.get("content", "")

        if role not in ("user", "assistant"):
            continue

        if not isinstance(text, str) or not text.strip():
            continue

        contents.append({
            "role": "user" if role == "user" else "model",
            "parts": [{"text": text[:8000]}]
        })

    if not contents or contents[-1]["role"] != "user":
        return jsonify({
            "error": "The last message must be from the user."
        }), 400

    client = genai.Client(
    api_key=api_key.strip()
)

app.logger.info("Gemini client initialized successfully")
)

    models_to_try = [PRIMARY_MODEL]

    if FALLBACK_MODEL and FALLBACK_MODEL != PRIMARY_MODEL:
        models_to_try.append(FALLBACK_MODEL)

    last_error = None

    for model in models_to_try:
        # محاولتان عند الأخطاء المؤقتة، مع انتظار قصير.
        for attempt in range(2):
            try:
                reply = generate_reply(client, model, contents)

                app.logger.info(
                    "Gemini response generated successfully using %s",
                    model
                )

                return jsonify({
                    "reply": reply,
                    "model": model
                })

            except Exception as error:
                last_error = error
                error_text = str(error)

                app.logger.warning(
                    "Model %s failed on attempt %s: %s",
                    model,
                    attempt + 1,
                    error_text
                )

                # أخطاء النموذج أو الطلب غير الصالحة لن تستفيد
                # غالبًا من إعادة المحاولة.
                if any(code in error_text for code in (
                    "400", "401", "403", "404"
                )):
                    break

                if attempt == 0:
                    time.sleep(2)

    app.logger.exception(
        "All configured Gemini model attempts failed",
        exc_info=last_error
    )

    return jsonify({
        "error": (
            "ORVEXA AI is temporarily unavailable. "
            "Please try again shortly."
        )
    }), 502


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "10000"))
    )

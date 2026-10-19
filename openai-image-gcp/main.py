import base64
import json
import mimetypes
import os
import sys
import tempfile

from openai import OpenAI
from google.cloud import storage
import functions_framework

storage_client = storage.Client()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

IN_PREFIX = os.getenv("IN_PREFIX", "mosaic/")
OUT_PREFIX = os.getenv("OUT_PREFIX", "analysis/")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

@functions_framework.cloud_event
def analyze_image(cloud_event):
    data = cloud_event.data
    bucket_name = data["bucket"]
    object_name = data["name"]

    if not object_name.startswith(IN_PREFIX):
        return
    if not object_name.lower().endswith((".jpg", ".jpeg", ".png")):
        return

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(object_name)

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "input")
        blob.download_to_file_name(src_path)

        with open(src_path, "rb") as f:
            image_bytes = f.read()

    mime = mimetypes.guess_type(object_name)[0] or "image/jpeg"
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    response = client.responses.create(
        model="gpt-5-mini",
        input=[
            {
                "role": "system",
                "content": "あなたは画像要約アシスタントです。必ずJSONのみを返してください。"
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "この画像について description, tags, comment を日本語で作成してください。tags は3〜5個。"
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime};base64,{b64}"
                    }
                ]
            },
        ],
        text = {
            "format": {
                "type": "json_schema",
                "name": "photo_analysis",
                "schema": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 3,
                            "maxItems": 5
                        },
                        "comment": {"type": "string"},
                    },
                    "required": ["description", "tags", "comment"],
                    "additionalProperties": False
                },
            },
        }
    )

    result = json.loads(response.output_text)

    filename = os.path.basename(response.output_text)
    out_name = f"{OUT_PREFIX}{filename}.json"
    out_blob = bucket.blob(out_name)
    out_blob.upload_from_string(
        json.dumps(result, ensure_ascii=False),
        content_type="application/json; charset=utf-8"
    )

    print(f"Analyzed {object_name} -> {out_name}")


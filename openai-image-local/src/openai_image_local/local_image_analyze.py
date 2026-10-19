import base64
import json
import mimetypes
import os
import sys

from openai import OpenAI

def main():
    if len(sys.argv) < 2:
        print("使い方: python local_image_analyze.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("環境変数 OPENAI_API_KEY が設定されていません。")
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    client = OpenAI(api_key=api_key)

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
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

import os
import json
import urllib.request
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"

headers = {
    "Content-Type": "application/json",
}

body = json.dumps({
    "contents": [{
        "parts": [{"text": "What is a neural network in one sentence?"}]
    }]
}).encode("utf-8")

req = urllib.request.Request(url, data=body, headers=headers, method="POST")

try:
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
        print(result["candidates"][0]["content"]["parts"][0]["text"])
except urllib.error.HTTPError as e:
    print(f"Lỗi HTTP {e.code}: {e.reason}")
    print(e.read().decode())

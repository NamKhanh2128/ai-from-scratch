import os
from dotenv import load_dotenv
import anthropic

# Nạp biến môi trường từ file .env
load_dotenv()

# Tạo client với API key từ biến môi trường
client = anthropic.Anthropic(
    api_key=os.environ.get("_API_KEY")
)

# Gọi API
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=256,
    messages=[{"role": "user", "content": "What is a neural network in one sentence?"}]
)

# In câu trả lời
print(response.content[0].text)

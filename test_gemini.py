import os
from dotenv import load_dotenv
from google import genai

# Nạp biến môi trường từ file .env
load_dotenv()

# Client tự động lấy key từ biến môi trường GEMINI_API_KEY
client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What is a neural network in one sentence?"
)

print(response.text)


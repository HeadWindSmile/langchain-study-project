import requests

url = "http://127.0.0.1:8000/ask"

payload = {
    "question": "企业知识库助手为什么需要 RAG？"
}

response = requests.post(url, json=payload, timeout=60)

print(response.status_code)
print(response.json())
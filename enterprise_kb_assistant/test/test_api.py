import requests
import json


# url = "http://127.0.0.1:8000/ask"
#
# payload = {
#     "question": "企业知识库助手为什么需要 RAG？"
# }
#
# response = requests.post(url, json=payload, timeout=60)
#
# print(response.status_code)
# print(response.json())





url = "http://127.0.0.1:8000/ask/stream"

headers = {
    "Content-Type": "application/json",
    "X-API-Key": "dev-secret-key",
}

payload = {
    "question": "RAG 的核心流程是什么？"
}

with requests.post(url, headers=headers, json=payload, stream=True, timeout=120) as resp:
    print("status:", resp.status_code)

    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue

        print(line)

        if line.startswith("data: "):
            raw = line.removeprefix("data: ")
            try:
                data = json.loads(raw)
                print("parsed:", data)
            except json.JSONDecodeError:
                pass
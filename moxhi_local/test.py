import requests

response = requests.post(
    "http://127.0.0.1:8090/translate",
    json={
        "text": "这是一个测试。今天的天气很好。"
    },
)

print(response.status_code)
print(response.json()["text"])
print(response.json()["stats"])
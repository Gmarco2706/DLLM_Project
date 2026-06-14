import requests
import json

API_KEY = "sk_3d7615d1cc7c9f4df4ae20eaafdc034a"
API_URL = "https://api.inceptionlabs.ai/v1/chat/completions"

response = requests.post(
    API_URL,
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    },
    json={
        "model": "mercury-2",
        "temperature": 0.2,
        "messages": [{"role": "user", "content": "Test prompt. Reply with 1"}]
    },
    timeout=60
)

print(f"Status Code: {response.status_code}")
try:
    print(f"JSON Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Raw Response: {response.text}")

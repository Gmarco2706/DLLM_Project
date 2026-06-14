import requests
import json
import time

API_KEYS = [
    "sk_3d7615d1cc7c9f4df4ae20eaafdc034a",
    "sk_4a650cc697f305c6c3e47827a07e4f5d",
    "sk_598ba735e556ac6804bf8faf05d7d406",
    "sk_a32ae2ed35c1b15bb4522cb6b9df882f",
]
API_URL = "https://api.inceptionlabs.ai/v1/chat/completions"

for key in API_KEYS:
    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "mercury-2",
            "temperature": 0.2,
            "messages": [{"role": "user", "content": "Test prompt. Reply with 1"}]
        },
        timeout=60
    )
    print(f"Key: {key[:8]}... Status: {response.status_code}")
    try:
        print(f"Response: {response.json().get('error', {}).get('message', 'Success')}")
    except Exception as e:
        pass
    time.sleep(1)

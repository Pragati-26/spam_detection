import requests

url = "http://127.0.0.1:5000/predict"
data = {"message": "You have won $1000! Click the link now."}

try:
    response = requests.post(url, json=data)
    print("Status Code:", response.status_code)
    print("Raw Response:", response.text)

    if response.headers.get("Content-Type") == "application/json":
        print("Prediction:", response.json())
    else:
        print("Non-JSON response received.")
except requests.exceptions.RequestException as e:
    print("Request failed:", e)

import requests

url = "http://127.0.0.1:5000/analyze"
file_path = "dataset/ferrugem/1.jpg"

with open(file_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)
    print(response.json())

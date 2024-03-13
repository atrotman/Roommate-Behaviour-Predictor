import requests

url = 'http://127.0.0.1:5000/predict-file'
files = {'file': open('Mac ML - test data v1.xlsx', 'rb')}

response = requests.post(url, files=files)
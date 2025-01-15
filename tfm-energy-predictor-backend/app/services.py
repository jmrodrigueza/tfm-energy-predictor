import json
import os

import requests

# For Docker file : export API_TOKEN="your_api_token_here"
# Get the API token from the environment variable
API_TOKEN = os.getenv('API_TOKEN')
HUGGING_API_URL = "https://josemanuelr-tfm-energy-predictor.hf.space"
# Define your proxy settings
proxies = None


def hugging_face_req(req, req_path: str):
    try:
        headers = {
            "Authorization": f"Bearer {API_TOKEN}"
        }
        print(HUGGING_API_URL + f'/{req_path}')
        print(headers)
        print(json.loads(req.json()))
        response = requests.post(HUGGING_API_URL + f'/{req_path}', json=json.loads(req.json()), headers=headers,
                                 proxies=proxies)
        if response.status_code == 200:
            response = response.json()
        else:
            response = {'message': 'Error: data not available!', 'status': response.status_code}
    except requests.RequestException as e:
        response = {'message': str(e), 'status': 500}
    return response

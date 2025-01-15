import requests
import time

# Get your own API key at https://opendata.aemet.es/centrodedescargas/altaUsuario
api_key = '_your_own_api_token_'
base_url = 'https://opendata.aemet.es/opendata/api'


def do_request(endpoint_url, output_file):
    """
    Performs a GET request to the API and stores the content in a JSON file.

    Args:
        endpoint_url (str): URL of the endpoint to request.
        output_file (str): Path to the file where the content will be stored.

    Returns:
        None
    """

    url = f'{base_url}/{endpoint_url}/?api_key={api_key}'

    response = requests.get(url)

    if response.status_code == 200:
        content_json = response.json()

        if content_json["estado"] == 200:
            data_url = content_json["datos"]
            meta_data_url = content_json["metadatos"]
            print(f'  Downloading from "{data_url}"')
            print(f'  See metadata at "{meta_data_url}"')
            time.sleep(0.5)
            data_response = requests.get(data_url, stream=True)
            if data_response.status_code == 200:
                with open(output_file, 'wb') as fichero:
                    for chunk in data_response.iter_content(1024):
                        fichero.write(chunk)
                print(f'  Content stored in {output_file}.')
            else:
                print(f'  Error downloading the data file: {data_response.status_code}')
        else:
            print(f'  Error requesting data URL: error_code={content_json["estado"]}')
    else:
        print(f'  Error {response.status_code}: {response.text}')
import urllib.parse
import requests

'''
TODO:
- Improve error handling: https://stackoverflow.com/questions/16511337/correct-way-to-try-except-using-python-requests-module
- Add Dataset class (perhaps only for images for now, and based on releases)
'''


class SegmentsClient:
    def __init__(self, api_key, api_url='https://api.segments.ai/'):
        self.api_key = api_key
        self.api_url = api_url

        r = self.get('/api_status/?lib_version=0.1')
        if r.status_code == 200:
            print('Initialized successfully.')
        elif r.status_code == 426:
            print('Please upgrade: pip install segments-ai --upgrade')
        else:
            print('Something went wrong. Did you use the right api key?')

    def get_sample(self, uuid):
        r = self.get(f'/samples/{uuid}')
        return r.json()

    def add_sample(self, dataset, name, attributes):
        payload = {
            'name': name,
            'attributes': attributes
        }
        r = self.post(f'/datasets/{dataset}/samples/', payload)
        print(r.status_code)
        return r.json()

    def get_label(self, uuid, task_name):
        r = self.get(f'/labels/{uuid}/{task_name}')
        return r.json()

    def add_label(self, sample_uuid, task_name, attributes):
        payload = {
            'attributes': attributes
        }
        r = self.put(f'/labels/{sample_uuid}/{task_name}/', payload)
        return r.json()

    def get_dataset(self, dataset):
        r = self.get(f'/datasets/{dataset}')
        return r.json()

    def get_samples(self, dataset):
        r = self.get(f'/datasets/{dataset}/samples')
        return r.json()

    def upload_asset(self, file, filename):
        r = self.post('/assets/', {'filename': filename})
        response_aws = self._upload_to_aws(file.getvalue(), r.json()['presignedPostFields'])
        return r.json()

    def _get_auth_header(self):
        if self.api_key:
            return {'Authorization': f'APIKey {self.api_key}'}
        else:
            return None

    def get(self, endpoint, auth=True):
        headers = self._get_auth_header() if auth else None
        # try:
        r = requests.get(urllib.parse.urljoin(self.api_url, endpoint),
                         headers=headers)
        # r.raise_for_status()
        # except requests.exceptions.ConnectionError as errc:
        #     print("Error Connecting:", errc)
        # except requests.exceptions.Timeout as errt:
        #     print("Timeout Error:", errt)

        return r

    def post(self, endpoint, data=None, auth=True):
        headers = self._get_auth_header() if auth else None
        return requests.post(urllib.parse.urljoin(self.api_url, endpoint),
                             json=data,  # data=data
                             headers=headers)

    def put(self, endpoint, data=None, auth=True):
        headers = self._get_auth_header() if auth else None
        return requests.put(urllib.parse.urljoin(self.api_url, endpoint),
                             json=data,  # data=data
                             headers=headers)

    def patch(self, endpoint, data=None, auth=True):
        headers = self._get_auth_header() if auth else None
        return requests.patch(urllib.parse.urljoin(self.api_url, endpoint),
                              json=data,  # data=data
                              headers=headers)

    @staticmethod
    def _upload_to_aws(file, aws_fields):
        files = {'file': file}
        r = requests.post(aws_fields['url'],
                                 files=files,
                                 data=aws_fields['fields'])
        return r
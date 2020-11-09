import urllib.parse
import requests

'''
TODO:
- Improve error handling
'''

class SegmentsClient:
    def __init__(self, api_key, api_url='https://api.segments.ai/'):
        self.api_key = api_key
        self.api_url = api_url

        r = self.get('/api_status/?lib_version=0.1')
        if r.status_code == 200:
            print('Initialized successfully.')
        elif r.status_code == 426:
            print('Please upgrade: pip install segments-ai --upgrade') # TODO: also return this in the server error message
        else:
            raise Exception('Something went wrong. Did you use the right api key?')

    ############
    # Datasets #
    ############
    def get_datasets(self, user=None):
        if user is not None:
            r = self.get('/users/{}/datasets/'.format(user))
        else:
            r = self.get('/user/datasets/')
        return r.json()

    def get_dataset(self, dataset_identifier):
        r = self.get('/datasets/{}/'.format(dataset_identifier))
        return r.json()

    def add_dataset(self, name, description, category='other', public=False, readme=''):
        payload = {
            'name': name,
            'description': description,
            'category': category,
            'public': public,
            'readme': readme,
            'data_type': 'IMAGE'
        }
        r = self.post('/user/datasets/', payload)
        return r.json()

    def delete_dataset(self, dataset_identifier):
        r = self.delete('/datasets/{}/'.format(dataset_identifier))

    ###########
    # Samples #
    ###########
    def get_samples(self, dataset):
        r = self.get('/datasets/{}/samples/'.format(dataset))
        return r.json()

    def get_sample(self, uuid):
        r = self.get('/samples/{}/'.format(uuid))
        return r.json()

    def add_sample(self, dataset, name, attributes):
        payload = {
            'name': name,
            'attributes': attributes
        }
        r = self.post('/datasets/{}/samples/'.format(dataset), payload)
        print('Uploaded ' + name)
        return r.json()

    def delete_sample(self, uuid):
        r = self.delete('/samples/{}/'.format(uuid))

    ##########
    # Labels #
    ##########
    def get_label(self, sample_uuid, task_name):
        r = self.get('/labels/{}/{}/'.format(sample_uuid, task_name))
        return r.json()

    def add_label(self, sample_uuid, task_name, attributes, label_status='PRELABELED'):
        payload = {
            'label_status': label_status,
            'attributes': attributes
        }
        r = self.put('/labels/{}/{}/'.format(sample_uuid, task_name), payload)
        return r.json()

    def delete_label(self, sample_uuid, task_name):
        r = self.delete('/labels/{}/{}/'.format(sample_uuid, task_name))

    ############
    # Releases #
    ############
    def get_releases(self, dataset):
        r = self.get('/datasets/{}/releases/'.format(dataset))
        return r.json()

    def get_release(self, dataset, release):
        r = self.get('/datasets/{}/releases/{}/'.format(dataset, release))
        return r.json()

    def add_release(self, dataset, name, description=''):
        payload = {
            'name': name,
            'description': description
        }
        r = self.post('/datasets/{}/releases/'.format(dataset), payload)
        return r.json()

    def delete_release(self, dataset, release):
        r = self.delete('/datasets/{}/releases/{}/'.format(dataset, release))

    ##########
    # Assets #
    ##########
    def upload_asset(self, file, filename):
        r = self.post('/assets/', {'filename': filename})
        response_aws = self._upload_to_aws(file, r.json()['presignedPostFields'])
        return r.json()    

    # Error handling: https://stackoverflow.com/questions/16511337/correct-way-to-try-except-using-python-requests-module

    def _get_auth_header(self):
        if self.api_key:
            return {'Authorization': 'APIKey {}'.format(self.api_key)}
        else:
            return None

    def get(self, endpoint, auth=True):
        headers = self._get_auth_header() if auth else None

        try:
            r = requests.get(urllib.parse.urljoin(self.api_url, endpoint),
                            headers=headers)
            r.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            print('{} | {}'.format(errh, r.json()))

        return r

    def post(self, endpoint, data=None, auth=True):
        headers = self._get_auth_header() if auth else None

        try:
            r = requests.post(urllib.parse.urljoin(self.api_url, endpoint),
                                json=data,  # data=data
                                headers=headers)
            r.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            print('{} | {}'.format(errh, r.json()))

        return r

    def put(self, endpoint, data=None, auth=True):
        headers = self._get_auth_header() if auth else None

        try:
            r = requests.put(urllib.parse.urljoin(self.api_url, endpoint),
                                json=data,  # data=data
                                headers=headers)
            r.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            print('{} | {}'.format(errh, r.json()))

        return r

    def patch(self, endpoint, data=None, auth=True):
        headers = self._get_auth_header() if auth else None

        try:
            r = requests.patch(urllib.parse.urljoin(self.api_url, endpoint),
                                json=data,  # data=data
                                headers=headers)
            r.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            print('{} | {}'.format(errh, r.json()))

        return r

    def delete(self, endpoint, data=None, auth=True):
        headers = self._get_auth_header() if auth else None

        try:
            r = requests.delete(urllib.parse.urljoin(self.api_url, endpoint),
                                json=data,  # data=data
                                headers=headers)
            r.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            print('{}'.format(errh))

        return r

    @staticmethod
    def _upload_to_aws(file, aws_fields):
        files = {'file': file}
        r = requests.post(aws_fields['url'],
                                 files=files,
                                 data=aws_fields['fields'])
        return r
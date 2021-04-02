import urllib.parse
import requests


class SegmentsClient:
    """SegmentsClient class.

    Args:
        api_key (str): Your Segments.ai API key.
        api_url (str, optional): URL of the Segments.ai API.

    Attributes:
        api_key (str): Your Segments.ai API key.
        api_url (str): URL of the Segments.ai API.

    """

    def __init__(self, api_key, api_url='https://api.segments.ai/'):
        self.api_key = api_key
        self.api_url = api_url

        r = self.get('/api_status/?lib_version=0.31')
        if r.status_code == 200:
            print('Initialized successfully.')
        elif r.status_code == 426:
            pass
        else:
            raise Exception('Something went wrong. Did you use the right api key?')

    ############
    # Datasets #
    ############
    def get_datasets(self, user=None):
        """Get a list of datasets.

        Args:
            user (str, optional): The user for which to get the datasets. Leave empty to get datasets of current user. Defaults to None.

        Returns:
            list: a list of dictionaries representing the datasets.
        """

        if user is not None:
            r = self.get('/users/{}/datasets/'.format(user))
        else:
            r = self.get('/user/datasets/')
        return r.json()

    def get_dataset(self, dataset_identifier):
        """Get a dataset.

        Args:
            dataset_identifier (str): The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.

        Returns:
            dict: a dictionary representing the dataset.
        """

        r = self.get('/datasets/{}/'.format(dataset_identifier))
        return r.json()

    def add_dataset(self, name, description='', task_type='segmentation-bitmap', task_attributes=None, category='other', public=False, readme=''):
        """Add a dataset.

        Args:
            name (str): The dataset name. Example: flowers.
            description (str, optional): The dataset description. Defaults to ''.
            task_type (str, optional): The dataset's task type. Defaults to 'segmentation-bitmap'.
            task_attributes (dict, optional): The dataset's task attributes. Defaults to None.
            category (str, optional): The dataset category. Defaults to 'other'.
            public (bool, optional): The dataset visibility. Defaults to False.
            readme (str, optional): The dataset readme. Defaults to ''.

        Returns:
            dict: a dictionary representing the newly created dataset.
        """

        if task_attributes is None:
            task_attributes = {
                "format_version": "0.1",
                "categories": [
                    {
                    "id": 0,
                    "name": "object"
                    }
                ]
            }

        payload = {
            'name': name,
            'description': description,
            'task_type': task_type,
            'task_attributes': task_attributes,
            'category': category,
            'public': public,
            'readme': readme,
            'data_type': 'IMAGE'
        }
        r = self.post('/user/datasets/', payload)
        return r.json()

    def delete_dataset(self, dataset_identifier):
        """Delete a dataset.

        Args:
            dataset_identifier (str): The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
        """

        r = self.delete('/datasets/{}/'.format(dataset_identifier))

    ###########
    # Samples #
    ###########
    def get_samples(self, dataset, per_page=1000, page=1):
        """Get the samples in a dataset.

        Args:
            dataset (str): The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
            per_page (int, optional): Pagination parameter indicating the maximum number of samples to return. Defaults to 1000.
            page (int, optional): Pagination parameter indicating the page to return. Defaults to 1.

        Returns:
            list: a list of dictionaries representing the samples.
        """

        r = self.get('/datasets/{}/samples/?per_page={}&page={}'.format(dataset, per_page, page))
        return r.json()

    def get_sample(self, uuid):
        """Get a sample.

        Args:
            uuid (str): The sample uuid.

        Returns:
            dict: a dictionary representing the sample.
        """

        r = self.get('/samples/{}/'.format(uuid))
        return r.json()

    def add_sample(self, dataset, name, attributes, metadata=None, priority=None):
        """Add a sample to a dataset.

        Args:
            dataset (str): The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
            name (str): The name of the sample.
            attributes (dict): The sample attributes. Please refer to the online documentation.
            metadata (dict, optional): Any sample metadata. Example: {'weather': 'sunny', 'camera_id': 3}.
            priority (float, optional): Priority in the labeling queue. Samples with higher values will be labeled first. Default is 0.

        Returns:
            dict: a dictionary representing the newly created sample.
        """

        payload = {
            'name': name,
            'attributes': attributes,
        }

        if metadata is not None:
            payload['metadata'] = metadata

        if priority is not None:
            payload['priority'] = priority

        r = self.post('/datasets/{}/samples/'.format(dataset), payload)
        print('Added ' + name)
        return r.json()

    def update_sample(self, uuid, name=None, attributes=None, metadata=None, priority=None):
        """Update a sample.

        Args:
            uuid (str): The sample uuid.
            name (str, optional): The name of the sample.
            attributes (dict, optional): The sample attributes. Please refer to the online documentation.
            metadata (dict, optional): Any sample metadata. Example: {'weather': 'sunny', 'camera_id': 3}.
            priority (float, optional): Priority in the labeling queue. Samples with higher values will be labeled first. Default is 0.

        Returns:
            dict: a dictionary representing the updated sample.
        """

        payload = {}

        if name is not None:
            payload['name'] = name

        if attributes is not None:
            payload['attributes'] = attributes

        if metadata is not None:
            payload['metadata'] = metadata

        if priority is not None:
            payload['priority'] = priority

        r = self.patch('/samples/{}/'.format(uuid), payload)
        print('Updated ' + uuid)
        return r.json()

    def delete_sample(self, uuid):
        """Delete a sample.

        Args:
            uuid (str): The sample uuid.
        """

        r = self.delete('/samples/{}/'.format(uuid))

    ##########
    # Labels #
    ##########
    def get_label(self, sample_uuid, labelset):
        """Get a label.

        Args:
            sample_uuid (str): The sample uuid.
            labelset (str): The labelset this label belongs to.

        Returns:
            dict: a dictionary representing the label.
        """

        r = self.get('/labels/{}/{}/'.format(sample_uuid, labelset))
        return r.json()

    def add_label(self, sample_uuid, labelset, attributes, label_status='PRELABELED', score=None):
        """Add a label to a sample.

        Args:
            sample_uuid (str): The sample uuid.
            labelset (str): The labelset this label belongs to.
            attributes (dict): The label attributes. Please refer to the online documentation.
            label_status (str, optional): The label status. Defaults to 'PRELABELED'.

        Returns:
            dict: a dictionary representing the newly created label.
        """

        payload = {
            'label_status': label_status,
            'attributes': attributes
        }

        if score is not None:
            payload['score'] = score

        r = self.put('/labels/{}/{}/'.format(sample_uuid, labelset), payload)
        return r.json()

    def update_label(self, sample_uuid, labelset, attributes=None, label_status='PRELABELED', score=None):
        """Update a label.

        Args:
            sample_uuid (str): The sample uuid.
            labelset (str): The labelset this label belongs to.
            attributes (dict): The label attributes. Please refer to the online documentation.
            label_status (str, optional): The label status. Defaults to 'PRELABELED'.

        Returns:
            dict: a dictionary representing the updated label.
        """

        payload = {}

        if attributes is not None:
            payload['attributes'] = attributes

        if label_status is not None:
            payload['label_status'] = label_status

        if score is not None:
            payload['score'] = score

        r = self.patch('/labels/{}/{}/'.format(sample_uuid, labelset), payload)
        return r.json()

    def delete_label(self, sample_uuid, labelset):
        """Delete a label.

        Args:
            sample_uuid (str): The sample uuid.
            labelset (str): The labelset this label belongs to.
        """

        r = self.delete('/labels/{}/{}/'.format(sample_uuid, labelset))

    ############
    # Releases #
    ############
    def get_releases(self, dataset):
        """Get the releases in a dataset.

        Args:
            dataset (str): The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.

        Returns:
            list: a list of dictionaries representing the releases.
        """

        r = self.get('/datasets/{}/releases/'.format(dataset))
        return r.json()

    def get_release(self, dataset, release):
        """Get a release.

        Args:
            dataset (str): The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
            release (str): The release name.

        Returns:
            dict: a dictionary representing the release.
        """

        r = self.get('/datasets/{}/releases/{}/'.format(dataset, release))
        return r.json()

    def add_release(self, dataset, name, description=''):
        """Add a release to a dataset.

        Args:
            dataset (str): The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
            name (str): The name of the release.
            description (str, optional): The release description.

        Returns:
            dict: a dictionary representing the newly created release.
        """

        payload = {
            'name': name,
            'description': description
        }
        r = self.post('/datasets/{}/releases/'.format(dataset), payload)
        return r.json()

    def delete_release(self, dataset, release):
        """Delete a release.

        Args:
            dataset (str): The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
            release (str): The release name.
        """

        r = self.delete('/datasets/{}/releases/{}/'.format(dataset, release))

    ##########
    # Assets #
    ##########
    def upload_asset(self, file, filename):
        """Upload an asset.

        Args:
            file (object): A file object.
            filename (str): The file name.

        Returns:
            dict: a dictionary representing the uploaded file.
        """

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
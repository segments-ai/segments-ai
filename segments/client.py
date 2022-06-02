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

        # https://realpython.com/python-requests/#performance
        # https://stackoverflow.com/questions/21371809/cleanly-setting-max-retries-on-python-requests-get-or-post-method
        # https://stackoverflow.com/questions/23013220/max-retries-exceeded-with-url-in-requests
        self.api_session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(max_retries=3)
        self.api_session.mount('http://', adapter)
        self.api_session.mount('https://', adapter)

        self.s3_session = requests.Session()
        self.s3_session.mount('http://', adapter)
        self.s3_session.mount('https://', adapter)

        r = self.get('/api_status/?lib_version=0.73')
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

    def add_dataset(self, name, description='', task_type='segmentation-bitmap', task_attributes=None, category='other', public=False, readme='', enable_skip_labeling=True, enable_skip_reviewing=False, enable_ratings=False, organization=None):
        """Add a dataset.

        Args:
            name (str): The dataset name. Example: flowers.
            description (str, optional): The dataset description. Defaults to ''.
            task_type (str, optional): The dataset's task type. One of 'segmentation-bitmap', 'segmentation-bitmap-highres', 'vector', 'bboxes', 'keypoints'. Defaults to 'segmentation-bitmap', 'pointcloud-segmentation', 'pointcloud-detection'.
            task_attributes (dict, optional): The dataset's task attributes. Defaults to None.
            category (str, optional): The dataset category. Defaults to 'other'.
            public (bool, optional): The dataset visibility. Defaults to False.
            readme (str, optional): The dataset readme. Defaults to ''.
            enable_skip_labeling (bool, optional): Enable the skip button in the labeling workflow. Defaults to True.
            enable_skip_reviewing (bool, optional): Enable the skip button in the reviewing workflow. Defaults to False.
            enable_ratings (bool, optional): Enable star-ratings for labeled images. Defaults to False.
            organization (str, optional): The username of the organization for which this dataset should be created. None will create a dataset for the current user. Defaults to None.

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
            'enable_skip_labeling': enable_skip_labeling,
            'enable_skip_reviewing': enable_skip_reviewing,
            'enable_ratings': enable_ratings,
            'data_type': 'IMAGE'
        }

        endpoint = '/user/datasets/'
        if organization is not None:
            endpoint = '/organizations/{}/datasets/'.format(organization)

        r = self.post(endpoint, payload)
        return r.json()

    def update_dataset(self, dataset_identifier, description=None, task_type=None, task_attributes=None, category=None, public=None, readme=None, enable_skip_labeling=None, enable_skip_reviewing=None, enable_ratings=None):
        """Update a dataset.

        Args:
            dataset_identifier (str): The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
            description (str, optional): The dataset description.
            task_type (str, optional): The dataset's task type. One of 'segmentation-bitmap', 'segmentation-bitmap-highres', 'vector', 'bboxes', 'keypoints', 'pointcloud-segmentation', 'pointcloud-detection'.
            task_attributes (dict, optional): The dataset's task attributes.
            category (str, optional): The dataset category.
            public (bool, optional): The dataset visibility.
            readme (str, optional): The dataset readme.
            enable_skip_labeling (bool, optional): Enable the skip button in the labeling workflow.
            enable_skip_reviewing (bool, optional): Enable the skip button in the reviewing workflow.
            enable_ratings: Enable star-ratings for labeled images.

        Returns:
            dict: a dictionary representing the updated dataset.
        """

        payload = {}

        if description is not None:
            payload['description'] = description

        if task_type is not None:
            payload['task_type'] = task_type

        if task_attributes is not None:
            payload['task_attributes'] = task_attributes

        if category is not None:
            payload['category'] = category

        if public is not None:
            payload['public'] = public

        if readme is not None:
            payload['readme'] = readme

        if enable_skip_labeling is not None:
            payload['enable_skip_labeling'] = enable_skip_labeling

        if enable_skip_reviewing is not None:
            payload['enable_skip_reviewing'] = enable_skip_reviewing

        if enable_ratings is not None:
            payload['enable_ratings'] = enable_ratings

        r = self.patch('/datasets/{}/'.format(dataset_identifier), payload)
        print('Updated ' + dataset_identifier)
        return r.json()

    def delete_dataset(self, dataset_identifier):
        """Delete a dataset.

        Args:
            dataset_identifier (str): The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
        """

        r = self.delete('/datasets/{}/'.format(dataset_identifier))

    def add_dataset_collaborator(self, dataset_identifier, username, role='labeler'):
        """Add a collaborator to a dataset.

        Args:
            dataset_identifier (str): The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
            username (str): The username of the collaborator to be added.
            role (str, optional): The role of the collaborator to be added. One of labeler, reviewer, admin. Defaults to labeler.

        Returns:
            dict: a dictionary containing the newly added collaborator with its role.
        """
        payload = {
            'user': username,
            'role': role
        }
        r = self.post('/datasets/{}/collaborators/'.format(dataset_identifier), payload)
        return r.json()


    ###########
    # Samples #
    ###########
    def get_samples(self, dataset_identifier, name=None, label_status=None, metadata=None, sort='name', direction='asc', per_page=1000, page=1):
        """Get the samples in a dataset.

        Args:
            dataset_identifier (str): The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
            name (str, optional): Name to filter by. Defaults to None (no filtering).
            label_status (list, optional): List of label statuses to filter by. Defaults to None (no filtering).
            metadata (list, optional): List of 'key:value' metadata attributes to filter by. Defaults to None (no filtering).
            sort (str, optional): What to sort results by. One of 'name', 'created', 'priority'. Defaults to 'name'.
            direction (str, optional): Sorting direction. One of 'asc' (ascending) or 'desc' (descending). Defaults to 'asc'.
            per_page (int, optional): Pagination parameter indicating the maximum number of samples to return. Defaults to 1000.
            page (int, optional): Pagination parameter indicating the page to return. Defaults to 1.

        Returns:
            list: a list of dictionaries representing the samples.
        """

        # pagination
        query_string = '?per_page={}&page={}'.format(per_page, page)

        # filter by name
        if name is not None:
            query_string += '&name__contains={}'.format(name)

        # filter by metadata
        if metadata is not None:
            if isinstance(metadata, str):
                metadata = [metadata]
            query_string += '&filters={}'.format(','.join(metadata))

        # filter by label status
        if label_status is not None:
            if isinstance(label_status, str):
                label_status = [label_status]
            assert isinstance(label_status, list)
            label_status = [status.upper() for status in label_status]
            query_string += '&labelset=ground-truth&label_status={}'.format(','.join(label_status))

        # sorting
        sort_dict = { 'name': 'name', 'created': 'created_at', 'priority': 'priority' }
        assert sort in sort_dict
        assert direction in ['asc', 'desc']
        if sort != 'name':            
            direction_str = '' if direction == 'asc' else '-'
            sort_str = sort_dict[sort]
            query_string += '&sort={}{}'.format(direction_str, sort_str)

        r = self.get('/datasets/{}/samples/{}'.format(dataset_identifier, query_string))
        results = r.json()

        for result in results:
            result.pop('label', None)
        return results

    def get_sample(self, uuid, labelset=None):
        """Get a sample.

        Args:
            uuid (str): The sample uuid.
            labelset (str, optional): If defined, this additionally returns the label for the given labelset. Defaults to None. 

        Returns:
            dict: a dictionary representing the sample
        """

        query_string = '/samples/{}/'.format(uuid)

        if labelset is not None:
            query_string += '?labelset={}'.format(labelset)

        r = self.get(query_string)
        return r.json()

    def add_sample(self, dataset_identifier, name, attributes, metadata=None, priority=None, embedding=None):
        """Add a sample to a dataset.

        Args:
            dataset_identifier (str): The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
            name (str): The name of the sample.
            attributes (dict): The sample attributes. Please refer to the online documentation.
            metadata (dict, optional): Any sample metadata. Example: {'weather': 'sunny', 'camera_id': 3}.
            priority (float, optional): Priority in the labeling queue. Samples with higher values will be labeled first. Defaults to 0.
            embedding (array, optional): Embedding of this sample represented by an array of floats.

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

        if embedding is not None:
            payload['embedding'] = embedding

        r = self.post('/datasets/{}/samples/'.format(dataset_identifier), payload)
        print('Added ' + name)
        return r.json()


    def add_samples(self, dataset_identifier, samples):
        """Add samples to a dataset in bulk. When attempting to add samples which already exist, no error is thrown but the existing samples are returned without changes.

        Args:
            dataset_identifier (str): The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
            samples (list): A list of dicts with required `name`, `attributes` fields and optional `metadata`, `priority`, `embedding` fields. See `add_sample` for details.

        Returns:
            list: a list of dicts representing the newly created samples.
        """

        payload = samples

        r = self.post('/datasets/{}/samples_bulk/'.format(dataset_identifier), payload)
        return r.json()


    def update_sample(self, uuid, name=None, attributes=None, metadata=None, priority=None, embedding=None):
        """Update a sample.

        Args:
            uuid (str): The sample uuid.
            name (str, optional): The name of the sample.
            attributes (dict, optional): The sample attributes. Please refer to the online documentation.
            metadata (dict, optional): Any sample metadata. Example: {'weather': 'sunny', 'camera_id': 3}.
            priority (float, optional): Priority in the labeling queue. Samples with higher values will be labeled first. Default is 0.
            embedding (array, optional): Embedding of this sample represented by an array of floats.

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

        if embedding is not None:
            payload['embedding'] = embedding

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
    def get_label(self, sample_uuid, labelset='ground-truth'):
        """Get a label.

        Args:
            sample_uuid (str): The sample uuid.
            labelset (str): The labelset this label belongs to. Defaults to 'ground-truth'.

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
            score (float, optional): The label score. Defaults to None.

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
            score (float, optional): The label score. Defaults to None.

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


    #############
    # Labelsets #
    #############
    def get_labelsets(self, dataset_identifier):
        """Get the labelsets in a dataset.

        Args:
            dataset_identifier (str): The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.

        Returns:
            list: a list of dictionaries representing the labelsets.
        """

        r = self.get('/datasets/{}/labelsets/'.format(dataset_identifier))
        return r.json()

    def get_labelset(self, dataset_identifier, name):
        """Get a labelset.

        Args:
            dataset_identifier (str): The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
            name (str): The name of the labelset.

        Returns:
            dict: a dictionary representing the labelset.
        """

        r = self.get('/datasets/{}/labelsets/{}/'.format(dataset_identifier, name))
        return r.json()

    def add_labelset(self, dataset_identifier, name, description=''):
        """Add a labelset to a dataset.

        Args:
            dataset_identifier (str): The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
            name (str): The name of the labelset.
            description (str, optional): The labelset description.

        Returns:
            dict: a dictionary representing the labelset.
        """

        payload = {
            'name': name,
            'description': description,
            'attributes': '{}',
        }
        r = self.post('/datasets/{}/labelsets/'.format(dataset_identifier), payload)
        return r.json()

    def delete_labelset(self, dataset_identifier, name):
        """Delete a labelset.

        Args:
            dataset_identifier (str): The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
            name (str): The name of the labelset.
        """

        r = self.delete('/datasets/{}/labelsets/{}/'.format(dataset_identifier, name))

    ############
    # Releases #
    ############
    def get_releases(self, dataset_identifier):
        """Get the releases in a dataset.

        Args:
            dataset_identifier (str): The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.

        Returns:
            list: a list of dictionaries representing the releases.
        """

        r = self.get('/datasets/{}/releases/'.format(dataset_identifier))
        return r.json()

    def get_release(self, dataset_identifier, name):
        """Get a release.

        Args:
            dataset_identifier (str): The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
            name (str): The name of the release.

        Returns:
            dict: a dictionary representing the release.
        """

        r = self.get('/datasets/{}/releases/{}/'.format(dataset_identifier, name))
        return r.json()

    def add_release(self, dataset_identifier, name, description=''):
        """Add a release to a dataset.

        Args:
            dataset_identifier (str): The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
            name (str): The name of the release.
            description (str, optional): The release description.

        Returns:
            dict: a dictionary representing the newly created release.
        """

        payload = {
            'name': name,
            'description': description
        }
        r = self.post('/datasets/{}/releases/'.format(dataset_identifier), payload)
        return r.json()

    def delete_release(self, dataset_identifier, name):
        """Delete a release.

        Args:
            dataset_identifier (str): The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
            name (str): The name of the release.
        """

        r = self.delete('/datasets/{}/releases/{}/'.format(dataset_identifier, name))

    ##########
    # Assets #
    ##########
    def upload_asset(self, file, filename='label.png'):
        """Upload an asset.

        Args:
            file (object): A file object.
            filename (str, optional): The file name. Defaults to label.png.

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
            r = self.api_session.get(urllib.parse.urljoin(self.api_url, endpoint),
                            headers=headers)
            r.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            print('{} | {}'.format(errh, r.json()))

        return r

    def post(self, endpoint, data=None, auth=True):
        headers = self._get_auth_header() if auth else None

        try:
            r = self.api_session.post(urllib.parse.urljoin(self.api_url, endpoint),
                                json=data,  # data=data
                                headers=headers)
            r.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            print('{} | {}'.format(errh, r.json()))

        return r

    def put(self, endpoint, data=None, auth=True):
        headers = self._get_auth_header() if auth else None

        try:
            r = self.api_session.put(urllib.parse.urljoin(self.api_url, endpoint),
                                json=data,  # data=data
                                headers=headers)
            r.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            print('{} | {}'.format(errh, r.json()))

        return r

    def patch(self, endpoint, data=None, auth=True):
        headers = self._get_auth_header() if auth else None

        try:
            r = self.api_session.patch(urllib.parse.urljoin(self.api_url, endpoint),
                                json=data,  # data=data
                                headers=headers)
            r.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            print('{} | {}'.format(errh, r.json()))

        return r

    def delete(self, endpoint, data=None, auth=True):
        headers = self._get_auth_header() if auth else None

        try:
            r = self.api_session.delete(urllib.parse.urljoin(self.api_url, endpoint),
                                json=data,  # data=data
                                headers=headers)
            r.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            print('{}'.format(errh))

        return r

    def _upload_to_aws(self, file, aws_fields):
        files = {'file': file}
        r = self.s3_session.post(aws_fields['url'],
                                 files=files,
                                 data=aws_fields['fields'])
        return r
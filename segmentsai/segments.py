import os
import requests
from io import BytesIO

import json

import numpy as np
from PIL import Image

# from torch.utils.data import Dataset

# class SegmentsDataset(Dataset):
class SegmentsDataset():
    def __init__(self, username, project, api_key, api_url='https://api.segments.ai/', filter_by=None,
                 transform=None, cache_labels=False, cache_dir='.cache/'):
        self.username = username
        self.project = project
        self.transform = transform
        self.cache_dir = cache_dir
        self.cache_labels = cache_labels

        segments = SegmentsClient(api_key, api_url)
        self.segments = segments

        self.load_project(project, filter_by)

    def load_project(self, project, filter_by=None):
        # Get project info
        print(f'Initializing dataset. This may take a few seconds...')
        self.project_info = self.segments.get_project(self.username, project)
        # print(self.project_info)
        self.labels = [None, ] + self.project_info['label_taxonomy']

        # Setup cache
        self.project_cache_dir = os.path.join(self.cache_dir, self.project_info['owner']['username'],
                                              self.project_info['name'])
        if not os.path.exists(self.project_cache_dir):
            os.makedirs(self.project_cache_dir)

        # Get samples in project and filter them
        self.images = self.segments.get_images(self.username, project)

        self.images = [image for image in self.images if image['superpixel_status'] == "SUCCEEDED"]

        if filter_by is not None:
            self.images = [image for image in self.images if image['label_status'] in filter_by]

        print(f'Initialized dataset with {len(self)} images.')

    def _load_image(self, image_url, from_cache=False):
        # print(image_url)
        # if isinstance(image_url, str):
        filename = os.path.join(self.project_cache_dir, '-'.join(image_url.split('/')[3:]))
        if from_cache and os.path.exists(filename):
            image = Image.open(filename)
        else:
            response = requests.get(image_url)
            # print(image_url, response)
            image = Image.open(BytesIO(response.content))
            image.save(filename)
        return image, os.path.basename(filename)

    def _load_label_data(self, json_url, from_cache=False):
        # print(image_url)
        # if isinstance(image_url, str):
        filename = os.path.join(self.project_cache_dir, '-'.join(json_url.split('/')[3:]))
        if from_cache and os.path.exists(filename):
            label_data = json.load(filename)
        else:
            response = requests.get(json_url)
            # print(image_url, response)
            label_data = json.load(BytesIO(response.content))
            # TODO: save label_data to cache.
        return label_data

    @staticmethod
    def _extract_label(label):
        label = np.array(label)
        # label = np.array(label, copy=False) # TODO: check why this doesn't work
        label[:, :, 3] = 0
        instance_label = label.view(np.uint32).squeeze(2)

        instance_label = Image.fromarray(instance_label)
        return instance_label

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # image = self.images[index]
        image = self.segments.get_image(self.images[index]['uuid'])

        # print(image)

        image_rgb, cache_filename = self._load_image(image['regular_url'], from_cache=True)

        label, _ = self._load_image(image['label_bitmap_url'], from_cache=self.cache_labels)
        label_bitmap = self._extract_label(label)

        superpixel, _ = self._load_image(image['superpixel_bitmap_urls'][-1], from_cache=False)
        superpixel_bitmap = self._extract_label(superpixel)

        label_data = self._load_label_data(image['label_data_url'], from_cache=self.cache_labels)

        sample = {
            'uuid': image['uuid'],
            'filename': image['filename'],
            'cache_filename': cache_filename,
            # 'label_status': image['label_status'],
            'image': image_rgb,
            'label_bitmap': label_bitmap,
            'label_data': label_data,
            'superpixel_bitmap': superpixel_bitmap
        }

        # transform
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    # response = requests.get(image['superpixel_bitmap_urls'][0])
    # superpixel_map = np.array(Image.open(BytesIO(response.content)))
    # superpixel_map[:,:,3] = 0
    # superpixel_map = superpixel_map.view(np.uint32).squeeze(2)

class SegmentsClient:
    def __init__(self, api_key, api_url='https://api.segments.ai/'):
        self.api_url = api_url
        self.api_key = api_key

    def get_project(self, username, project):
        response = self.get('projects/{}/{}/'.format(username, project))
        return response.json()

    def post_project(self, name, description, labels):
        response = self.post('projects/', {'name': name, 'description': description, 'label_taxonomy': labels})
        return response.json()

    def get_images(self, username, project):
        response = self.get('projects/{}/{}/images'.format(username, project))
        # print('get_images()', response.json())
        return response.json()

    def get_image(self, image_id):
        response = self.get('images/{}'.format(image_id))
        return response.json()

    def _upload_to_aws(self, file, aws_fields):
        files = {'file': file}
        response = requests.post(aws_fields['url'],
                                 files=files,
                                 data=aws_fields['fields'])
        return response

    def upload_image(self, username, project, image=None, uri=None, filename=None):
        # image must be a numpy array

        if image is not None:
            assert uri is None and filename is not None

            # Step 1: request signed aws urls from segments
            response = self.post('assets/',
                                 {'filename': filename})
            # print(response.json())
            uuid = response.json()['uuid']
            asset_url = response.json()['url']

            # Step 2: post the image to aws
            file = BytesIO()
            image.save(file, 'png')
            response = self._upload_to_aws(file.getvalue(), response.json()['presignedPostFields'])
            # print(response)

            # Step 3: post the aws url to segments
            response = self.post(f'projects/{username}/{project}/images/',
                                 {'filename': filename,
                                  'url': asset_url})
            # print(response.json())
            return response.json()
        elif uri is not None:
            assert image is None

            # Post the uri directly to segments
            if filename is None:
                filename = os.path.basename(uri)
            response = self.post(f'projects/{username}/{project}/images/',
                                 {'filename': filename,
                                  'url': uri})
            # print(response.json())
        else:
            assert False

    def save_prediction(self, image_uuid, label, annotation_data=None):
        # label: np.int32 array
        # annotation_data: json that looks like this: [{'instanceId': 1, 'classId': 0}, {'instanceId': 2, 'classId': 0}]

        label2 = np.copy(label)
        label2 = label2[:, :, None].view(np.uint8)
        label2[:, :, 3] = 255
        label_file = BytesIO()
        Image.fromarray(label2).save(label_file, 'PNG')

        response = self.post(f'images/{image_uuid}/upload_prediction')
        res = self._upload_to_aws(label_file.getvalue(), response.json()['prediction_bitmap_url'])

        if annotation_data is not None:
            res = self._upload_to_aws(json.dumps(annotation_data), response.json()['prediction_data_url'])
        # print(response)

    def save_label(self, image_uuid, label, annotation_data=None):
        # label: np.int32 array
        # annotation_data: json that looks like this: [{'instanceId': 1, 'classId': 0}, {'instanceId': 2, 'classId': 0}]

        label2 = np.copy(label)
        label2 = label2[:, :, None].view(np.uint8)
        label2[:, :, 3] = 255
        label_file = BytesIO()
        Image.fromarray(label2).save(label_file, 'PNG')

        response = self.post(f'images/{image_uuid}/upload')
        res = self._upload_to_aws(label_file.getvalue(), response.json()['label_bitmap_url'])

        if annotation_data is not None:
            res = self._upload_to_aws(json.dumps(annotation_data), response.json()['label_data_url'])
        # print(response)

    def get_auth_header(self):
        if self.api_key:
            return {'Authorization': f'APIKey {self.api_key}'}
        else:
            return None

    def get(self, endpoint, auth=True):
        headers = self.get_auth_header() if auth else None
        return requests.get(self.api_url + endpoint,
                            headers=headers)

    def post(self, endpoint, data=None, auth=True):
        headers = self.get_auth_header() if auth else None
        return requests.post(self.api_url + endpoint,
                             json=data,  # data=data
                             headers=headers)

    def patch(self, endpoint, data=None, auth=True):
        headers = self.get_auth_header() if auth else None
        return requests.patch(self.api_url + endpoint,
                              json=data,  # data=data
                              headers=headers)

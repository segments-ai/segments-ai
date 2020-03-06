import os
import requests
from io import BytesIO

import json

import numpy as np
from PIL import Image

# from torch.utils.data import Dataset

# class SegmentsDataset(Dataset):
class SegmentsDataset():
    def __init__(self, client, project, filter_by=None, transform=None):
        self.project = project
        self.transform = transform
        self.client = client
        self.load_project(project, filter_by)

    def load_project(self, project, filter_by=None):
        # Get project info
        print(f'Initializing dataset. This may take a few seconds...')
        self.project_info = self.client.get_project(project)
        # print(self.project_info)
        self.labels = [None, ] + self.project_info['label_taxonomy']

        # Get samples in project and filter them
        self.images = self.client.get_images(project)

        self.images = [image for image in self.images if image['superpixel_status'] == "SUCCEEDED"]

        if filter_by is not None:
            self.images = [image for image in self.images if image['label_status'] in filter_by]

        print(f'Initialized dataset with {len(self)} images.')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # image = self.images[index]
        sample = self.client.get_image(self.images[index]['uuid'], extract=True)

        # print(image)

        sample = {
            'uuid': sample['uuid'],
            'filename': sample['filename'],
            'cache_filename': sample['cache_filename'],
            'label_status': sample['label_status'],
            'image': sample['image'],
            'label': sample['label'],
            'label_data': sample['label_data'],
            # 'superpixel_bitmap': sample['superpixel_bitmap']
        }

        # transform
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    # response = requests.get(image['superpixel_bitmap_urls'][0])
    # superpixel_map = np.array(Image.open(BytesIO(response.content)))
    # superpixel_map[:,:,3] = 0
    # superpixel_map = superpixel_map.view(np.uint32).squeeze(2)

class SegmentsClient():
    def __init__(self, api_key, api_url='https://api.segments.ai/', cache_dir='.cache/', cache_labels=False,):
        # print('Initializing Segments client.')
        self.api_url = api_url
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.cache_labels = cache_labels

        # Setup cache
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        response = self.get('api_status/?lib_version=0.1')
        if (response.status_code == 200):
            print('Successfully initialized Segments client.')
        elif response.status_code == 426:
            print('Please upgrade: pip install segments-ai --upgrade')
        else:
            print('Something went wrong. Did you use the right api key?')

    def _load_image(self, image_url, from_cache=False):
        # print(image_url)
        # if isinstance(image_url, str):
        filename = os.path.join(self.cache_dir, '-'.join(image_url.split('/')[3:]))
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
        filename = os.path.join(self.cache_dir, '-'.join(json_url.split('/')[3:]))
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

    @staticmethod
    def _upload_to_aws(file, aws_fields):
        files = {'file': file}
        response = requests.post(aws_fields['url'],
                                 files=files,
                                 data=aws_fields['fields'])
        return response

    def get_project(self, project):
        # print(project)
        response = self.get(f'projects/{project}')
        return response.json()

    def post_project(self, name, description, labels):
        response = self.post('projects/', {'name': name, 'description': description, 'label_taxonomy': labels})
        return response.json()

    def get_images(self, project):
        response = self.get('projects/{}/images'.format(project))
        # print('get_images()', response.json())
        return response.json()

    def get_image(self, image_uuid, extract=True):
        response = self.get('images/{}'.format(image_uuid))
        result = response.json()
        # print(result)
        # result['segments_url'] = f'https://segments.ai/{"/".join(result["regular_url"].split("/")[3:])[:-7]}'
        if extract:
            image_rgb, cache_filename = self._load_image(result['regular_url'], from_cache=True)
            label_data = self._load_label_data(result['label_data_url'], from_cache=self.cache_labels)

            # print(label_data)
            if label_data['format_version'] == '0.1':
                label_data = label_data['annotations']
            else:
                assert False

            label, _ = self._load_image(result['label_bitmap_url'], from_cache=self.cache_labels)
            label_bitmap = self._extract_label(label)

            # superpixel, _ = self._load_image(image['superpixel_bitmap_urls'][-1], from_cache=False)
            # superpixel_bitmap = self._extract_label(superpixel)

            result['image'] = image_rgb
            result['label'] = label_bitmap
            result['label_data'] = label_data
            result['cache_filename'] = cache_filename
        return result

    def upload_image(self, project, image_numpy=None, image_url=None, filename=None):
        # image must be a numpy array
        # TODO: assert

        if image_numpy is not None:
            assert image_url is None and filename is not None

            # Step 1: request signed aws urls from segments
            response = self.post('assets/',
                                 {'filename': filename})
            # print(response.json())
            uuid = response.json()['uuid']
            asset_url = response.json()['url']

            # Step 2: post the image to aws
            file = BytesIO()
            image_numpy.save(file, 'png')
            response = self._upload_to_aws(file.getvalue(), response.json()['presignedPostFields'])
            # print(response)

            # Step 3: post the aws url to segments
            response = self.post(f'projects/{project}/images/',
                                 {'filename': filename,
                                  'public_url': asset_url})
            # print(response.json())
            result = response.json()
        elif image_url is not None:
            assert image_numpy is None

            # Post the uri directly to segments
            if filename is None:
                filename = os.path.basename(image_url)
            response = self.post(f'projects/{project}/images/',
                                 {'filename': filename,
                                  'public_url': image_url})
            result = response.json()
        else:
            assert False

        # result['segments_url'] = f'https://segments.ai/{project}/{result["uuid"]}'
        return result

    def upload_prediction(self, image_uuid, label, annotation_data=None):
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
            annotation_data = {
                'format_version': '0.1',
                'annotations': annotation_data
            }
            res = self._upload_to_aws(json.dumps(annotation_data), response.json()['prediction_data_url'])
        # print(response)

    def upload_label(self, image_uuid, label, annotation_data=None):
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
            annotation_data = {
                'format_version': '0.1',
                'annotations': annotation_data
            }
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

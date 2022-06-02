import os
import json
import requests
from urllib.parse import urlparse
from multiprocessing.pool import ThreadPool

from tqdm import tqdm

from .utils import load_image_from_url, load_label_bitmap_from_url, handle_exif_rotation

from PIL import Image

class SegmentsDataset():
    """SegmentsDataset class.

    Args:
        release_file (str or dict): Path to a release file, or a release dict resulting from client.get_release().
        labelset (str, optional): The labelset that should be loaded. Defaults to 'ground-truth'.
        filter_by (list, optional): A list of label statuses to filter by. Defaults to None.
        filter_by_metadata (dict, optional): a dict of metadata key:value pairs to filter by. Filters are ANDed together. Defaults to None.
        segments_dir (str, optional): The directory where the data will be downloaded to for caching. Set to None to disable caching. Defaults to 'segments'.
        preload (bool, optional): Whether the data should be pre-downloaded when the dataset is initialized. Ignored if segments_dir is None. Defaults to True. 
        s3_client (obj, optional): A boto3 S3 client, e.g. `s3_client = boto3.client("s3")`. Needs to be provided if your images are in a private S3 bucket. Defaults to None.

    """

    # https://stackoverflow.com/questions/682504/what-is-a-clean-pythonic-way-to-have-multiple-constructors-in-python
    def __init__(self, release_file, labelset='ground-truth', filter_by=None, filter_by_metadata=None, segments_dir='segments', preload=True, s3_client=None):
        self.labelset = labelset
        self.filter_by = [filter_by] if isinstance(filter_by, str) else filter_by
        if self.filter_by is not None:
            self.filter_by = [s.lower() for s in self.filter_by]
        self.filter_by_metadata = filter_by_metadata
        self.segments_dir = segments_dir
        self.caching_enabled = segments_dir is not None
        self.preload = preload
        self.s3_client = s3_client
        
        # if urlparse(release_file).scheme in ('http', 'https'): # If it's a url
        if isinstance(release_file, str): # If it's a file path
            with open(release_file) as f:
                self.release = json.load(f)
        else: # If it's a release object
            release_file = release_file['attributes']['url']
            content = requests.get(release_file)
            self.release = json.loads(content.content)        
        self.release_file = release_file

        self.dataset_identifier = '{}_{}'.format(self.release['dataset']['owner'], self.release['dataset']['name'])

        self.image_dir = None if segments_dir is None else os.path.join(segments_dir, self.dataset_identifier, self.release['name'])

        # First some checks
        if not self.labelset in [labelset['name'] for labelset in self.release['dataset']['labelsets']]:
            print('There is no labelset with name "{}".'.format(self.labelset))
            return

        self.task_type = self.release['dataset']['task_type']
        if self.task_type not in ['segmentation-bitmap', 'segmentation-bitmap-highres', 'vector', 'bboxes', 'keypoints', 'image-vector-sequence', 'pointcloud-detection', 'pointcloud-segmentation']:
            print('You can only create a dataset for tasks of type "segmentation-bitmap", "segmentation-bitmap-highres", "vector", "bboxes", "keypoints", "image-vector-sequence", "pointcloud-detection", or "pointcloud-segmentation" for now.')
            return
        
        self.load_dataset()

    def load_dataset(self):
        print('Initializing dataset...')
        
        # Setup cache
        if self.caching_enabled and not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
        
        # Load and filter the samples
        samples = self.release['dataset']['samples']
        if self.filter_by is not None:
            filtered_samples = []
            for sample in samples:
                if sample['labels'][self.labelset] is not None:
                    label_status = sample['labels'][self.labelset]['label_status'].lower()
                else:
                    label_status = 'unlabeled'

                if label_status in self.filter_by:
                    filtered_samples.append(sample)
            samples = filtered_samples

        if self.filter_by_metadata is not None:
            filtered_samples = []
            for sample in samples:
                if self.filter_by_metadata.items() <= sample['metadata'].items(): # https://stackoverflow.com/a/41579450/1542912
                    filtered_samples.append(sample)
            samples = filtered_samples

        self.samples = samples
            
        # # Preload all samples (sequentially)
        # for i in tqdm(range(self.__len__())):
        #     item = self.__getitem__(i)

        # To avoid memory overflow or "Too many open files" error when using tqdm in combination with multiprocessing.
        def _load_image(i):
            self.__getitem__(i)
            return i

        # Preload all samples (in parallel)
        # https://stackoverflow.com/questions/16181121/a-very-simple-multithreading-parallel-url-fetching-without-queue/27986480
        # https://stackoverflow.com/questions/3530955/retrieve-multiple-urls-at-once-in-parallel
        # https://github.com/tqdm/tqdm/issues/484#issuecomment-461998250
        num_samples = self.__len__()
        if self.caching_enabled and self.preload and self.task_type not in ['pointcloud-segmentation', 'pointcloud-detection']:
            print('Preloading all samples. This may take a while...')
            with ThreadPool(16) as pool:
                # r = list(tqdm(pool.imap_unordered(self.__getitem__, range(num_samples)), total=num_samples))
                r = list(tqdm(pool.imap_unordered(_load_image, range(num_samples)), total=num_samples))

        print('Initialized dataset with {} images.'.format(num_samples))

    def _load_image_from_cache(self, sample):
        sample_name = os.path.splitext(sample['name'])[0]
        image_url = sample['attributes']['image']['url']
        image_url_parsed = urlparse(image_url)
        url_extension = os.path.splitext(image_url_parsed.path)[1]
        # image_filename_rel = '{}{}'.format(sample['uuid'], url_extension)
        image_filename_rel = '{}{}'.format(sample_name, url_extension)

        if image_url_parsed.scheme == 's3':
            image = None
        else:
            if self.caching_enabled:
                image_filename = os.path.join(self.image_dir, image_filename_rel)
                if not os.path.exists(image_filename):
                    image = load_image_from_url(image_url, image_filename, self.s3_client)
                else:
                    image = Image.open(image_filename)
            else:
                image = load_image_from_url(image_url, self.s3_client)            

            image = handle_exif_rotation(image)

        return image, image_filename_rel

    def _load_segmentation_bitmap_from_cache(self, sample, labelset):
        sample_name = os.path.splitext(sample['name'])[0]
        label = sample['labels'][labelset]
        segmentation_bitmap_url = label['attributes']['segmentation_bitmap']['url']
        url_extension = os.path.splitext(urlparse(segmentation_bitmap_url).path)[1]

        if self.caching_enabled:
            # segmentation_bitmap_filename = os.path.join(self.image_dir, '{}{}'.format(label['uuid'], url_extension))
            segmentation_bitmap_filename = os.path.join(self.image_dir, '{}_label_{}{}'.format(sample_name, labelset, url_extension))            
            if not os.path.exists(segmentation_bitmap_filename):
                segmentation_bitmap = load_label_bitmap_from_url(segmentation_bitmap_url, segmentation_bitmap_filename)
            else:
                segmentation_bitmap = Image.open(segmentation_bitmap_filename)
        else:
            segmentation_bitmap = load_label_bitmap_from_url(segmentation_bitmap_url)            

        return segmentation_bitmap

    @property
    def categories(self):
        return self.release['dataset']['task_attributes']['categories']
        # categories = {}
        # for category in self.release['dataset']['labelsets'][self.labelset]['attributes']['categories']:
        #     categories[category['id']] = category['name']
        # return categories

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        if self.task_type in ['pointcloud-segmentation', 'pointcloud-detection', 'image-vector-sequence']:
            return sample
        
        # Load the image
        try:
            image, image_filename = self._load_image_from_cache(sample)
        except:
            print('Something went wrong loading sample {}:'.format(sample['name']), sample)
            raise

        item = {
            'uuid': sample['uuid'],
            'name': sample['name'],
            'file_name': image_filename,
            'image': image,
            'metadata': sample['metadata']
        }

        # Segmentation bitmap
        if self.task_type == 'segmentation-bitmap' or self.task_type == 'segmentation-bitmap-highres':
            # Load the label
            try:
                label = sample['labels'][self.labelset]
                segmentation_bitmap = self._load_segmentation_bitmap_from_cache(sample, self.labelset)
                attributes = label['attributes']
                annotations = attributes['annotations']
            except:
                segmentation_bitmap = attributes = annotations = None
            
            item.update({
                'segmentation_bitmap': segmentation_bitmap,
                'annotations': annotations,
                'attributes': attributes
            })

        # Vector labels
        elif self.task_type == 'vector' or self.task_type == 'bboxes' or self.task_type == 'keypoints':
            try:
                label = sample['labels'][self.labelset]
                attributes = label['attributes']
                annotations = attributes['annotations']
            except:
                attributes = annotations = None

            item.update({
                'annotations': annotations,
                'attributes': attributes
            })

        else:
            assert False

#         # transform
#         if self.transform is not None:
#             item = self.transform(item)

        return item
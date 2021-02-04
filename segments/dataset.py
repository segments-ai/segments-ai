import os
import json
import requests
from urllib.parse import urlparse
from multiprocessing.pool import ThreadPool
from tqdm import tqdm

from .utils import download_and_save_image, download_and_save_segmentation_bitmap, handle_exif_rotation

from PIL import Image

class SegmentsDataset():
    """SegmentsDataset class.

    Args:
        release_file (str or dict): Path to a release file, or a release dict resulting from client.get_release().
        labelset (str, optional): The labelset that should be loaded. Defaults to 'ground-truth'.
        filter_by (list, optional): A list of label statuses to filter by. Defaults to None.
        segments_dir (str, optional): The directory where the data will be downloaded to and stored. Defaults to 'segments'.

    """

    # https://stackoverflow.com/questions/682504/what-is-a-clean-pythonic-way-to-have-multiple-constructors-in-python
    def __init__(self, release_file, labelset='ground-truth', filter_by=None, segments_dir='segments'):
        self.labelset = labelset
        self.filter_by = [filter_by] if isinstance(filter_by, str) else filter_by
        if self.filter_by is not None:
            self.filter_by = [s.lower() for s in self.filter_by]
        self.segments_dir = segments_dir
        
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
        self.image_dir = os.path.join(segments_dir, self.dataset_identifier, self.release['name'])

        # First some checks
        if not self.labelset in [labelset['name'] for labelset in self.release['dataset']['labelsets']]:
            print('There is no labelset with name "{}".'.format(self.labelset))
            return

        self.task_type = self.release['dataset']['task_type']
        if self.task_type not in ['segmentation-bitmap', 'bboxes']:
            print('You can only create a dataset for tasks of type "segmentation-bitmap" or "bboxes" for now.')
            return
        
        self.load_dataset()

    def load_dataset(self):
        print('Initializing dataset. This may take a few seconds...')
        
        # Setup cache
        if not os.path.exists(self.image_dir):
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
        else:
            filtered_samples = samples

        self.samples = filtered_samples
            
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
        with ThreadPool(16) as pool:
            # r = list(tqdm(pool.imap_unordered(self.__getitem__, range(num_samples)), total=num_samples))
            r = list(tqdm(pool.imap_unordered(_load_image, range(num_samples)), total=num_samples))

        print('Initialized dataset with {} images.'.format(num_samples))

        
    def _load_image_from_cache(self, sample):
        sample_name = os.path.splitext(sample['name'])[0]
        image_url = sample['attributes']['image']['url']
        url_extension = os.path.splitext(urlparse(image_url).path)[1]
        # image_filename_rel = '{}{}'.format(sample['uuid'], url_extension)
        image_filename_rel = '{}{}'.format(sample_name, url_extension)
        image_filename = os.path.join(self.image_dir, image_filename_rel)

        if not os.path.exists(image_filename):
            download_and_save_image(image_url, image_filename)

        image = Image.open(image_filename)
        image = handle_exif_rotation(image)

        return image, image_filename_rel

    def _load_segmentation_bitmap_from_cache(self, sample, labelset):
        sample_name = os.path.splitext(sample['name'])[0]
        label = sample['labels'][labelset]
        segmentation_bitmap_url = label['attributes']['segmentation_bitmap']['url']
        url_extension = os.path.splitext(urlparse(segmentation_bitmap_url).path)[1]
        # segmentation_bitmap_filename = os.path.join(self.image_dir, '{}{}'.format(label['uuid'], url_extension))
        segmentation_bitmap_filename = os.path.join(self.image_dir, '{}_label_{}{}'.format(sample_name, labelset, url_extension))
        
        if not os.path.exists(segmentation_bitmap_filename):
            download_and_save_segmentation_bitmap(segmentation_bitmap_url, segmentation_bitmap_filename)

        segmentation_bitmap = Image.open(segmentation_bitmap_filename)

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
        }

        # Segmentation bitmap
        if self.task_type == 'segmentation-bitmap':
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

        # Bounding boxes
        elif self.task_type == 'bboxes':
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
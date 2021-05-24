import os
from io import BytesIO
import requests
import urllib.request
import json

import numpy as np
from PIL import Image, ExifTags

session = requests.Session()
adapter = requests.adapters.HTTPAdapter(max_retries=3)
session.mount('http://', adapter)
session.mount('https://', adapter)

def bitmap2file(bitmap, is_segmentation_bitmap=True):
    """Convert a label bitmap to a file with the proper format.

    Args:
        bitmap (np.uint32): A numpy array where each unique value represents an instance id.

    Returns:
        object: a file object.
    """

    # Convert bitmap to np.uint32, if it is not already
    if bitmap.dtype == 'uint32':
        pass
    elif bitmap.dtype == 'uint8':
        bitmap = np.uint32(bitmap)
    else:
        assert False

    if is_segmentation_bitmap:
        bitmap2 = np.copy(bitmap)
        bitmap2 = bitmap2[:, :, None].view(np.uint8)
        bitmap2[:, :, 3] = 255
    else:
        assert False
        
    f = BytesIO()
    Image.fromarray(bitmap2).save(f, 'PNG')
    f.seek(0)
    return f

def get_semantic_bitmap(instance_bitmap, annotations, id_increment=1):
    """Convert an instance bitmap and annotations dict into a segmentation bitmap.

    Args:
        instance_bitmap (np.uint32): A numpy array where each unique value represents an instance id.
        annotations (dict): An annotations dictionary.
        id_increment (int, optional): Increment the category ids with this number. Defaults to 1.

    Returns:
        np.uint32: a numpy array where each unique value represents a category id.
    """

    if instance_bitmap is None or annotations is None:
        return None

    instance2semantic = [0] * (max([a['id'] for a in annotations], default=0)+1)
    for annotation in annotations:
        instance2semantic[annotation['id']] = annotation['category_id'] + id_increment
    instance2semantic = np.array(instance2semantic)
        
    semantic_label = instance2semantic[np.array(instance_bitmap, np.uint32)]
    return semantic_label

def export_dataset(dataset, export_folder='.', export_format='coco-panoptic'):
    """Export a dataset to a different format.

    Args:
        dataset (dict): A dataset object, resulting from client.get_dataset().
        export_format (str, optional): The destination format. Can be 'coco-panoptic' (default), 'coco-instance', 'yolo.
    """

    print('Exporting dataset. This may take a while...')
    if export_format == 'coco-panoptic':
        from .export import export_coco_panoptic
        return export_coco_panoptic(dataset, export_folder)
    elif export_format == 'coco-instance':
        from .export import export_coco_instance
        return export_coco_instance(dataset, export_folder)
    elif export_format == 'yolo':
        from .export import export_yolo
        return export_yolo(dataset, export_folder)
    else:
        print('Supported export formats: coco-panoptic, coco-instance, yolo')
        return

def load_image_from_url(url, save_filename=None):
    """Load an image from url.

    Args:
        url (str): The image url.
        save_filename (str, optional): The filename to save to.

    Returns:
        PIL.Image: a PIL image.
    """
    image = Image.open(BytesIO(session.get(url).content))
    # urllib.request.urlretrieve(url, save_filename)

    if save_filename is not None:
        image.save(save_filename)

    return image

def load_label_bitmap_from_url(url, save_filename=None):
    """Load a label bitmap from url.

    Args:
        url (str): The label bitmap url.
        save_filename (str, optional): The filename to save to.

    Returns:
        np.uint32: a numpy np.uint32 array.
    """
    def extract_bitmap(bitmap):
        bitmap = np.array(bitmap)
        bitmap[:,:,3] = 0
        bitmap = bitmap.view(np.uint32).squeeze(2)
        return bitmap

    bitmap = Image.open(BytesIO(session.get(url).content))
    bitmap = extract_bitmap(bitmap)

    if save_filename is not None:
        Image.fromarray(bitmap).save(save_filename)

    return bitmap

def handle_exif_rotation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        exif=dict(image._getexif().items())
        if exif[orientation] == 3:
            image=image.transpose(Image.ROTATE_180)
        elif exif[orientation] == 6:
            image=image.transpose(Image.ROTATE_270)
        elif exif[orientation] == 8:
            image=image.transpose(Image.ROTATE_90)
        return image
    except (AttributeError, KeyError, IndexError):
        return image
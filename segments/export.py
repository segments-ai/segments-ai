# https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch/#coco-dataset-format

import os
import json
import random

from tqdm import tqdm
from shutil import copyfile

import numpy as np

from PIL import Image
from skimage import img_as_ubyte
from skimage.measure import regionprops
from skimage.segmentation import mark_boundaries
from pycocotools import mask

from .utils import get_semantic_bitmap

COLORMAP = [[0, 113, 188, 255], [216, 82, 24, 255], [236, 176, 31, 255], [125, 46, 141, 255], [118, 171, 47, 255], [76, 189, 237, 255], [161, 19, 46, 255], [255, 0, 0, 255], [255, 127, 0, 255], [190, 190, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [170, 0, 255, 255], [84, 84, 0, 255], [84, 170, 0, 255], [84, 255, 0, 255], [170, 84, 0, 255], [170, 170, 0, 255], [170, 255, 0, 255], [255, 84, 0, 255], [255, 170, 0, 255], [255, 255, 0, 255], [0, 84, 127, 255], [0, 170, 127, 255], [0, 255, 127, 255], [84, 0, 127, 255], [84, 84, 127, 255], [84, 170, 127, 255], [84, 255, 127, 255], [170, 0, 127, 255], [170, 84, 127, 255], [170, 170, 127, 255], [170, 255, 127, 255], [255, 0, 127, 255], [255, 84, 127, 255], [255, 170, 127, 255]]

# https://github.com/cocodataset/panopticapi/blob/master/panopticapi/utils.py
class IdGenerator():
    '''
    The class is designed to generate unique IDs that have meaningful RGB encoding.
    Given semantic category unique ID will be generated and its RGB encoding will
    have color close to the predefined semantic category color.
    The RGB encoding used is ID = R * 256 * G + 256 * 256 + B.
    Class constructor takes dictionary {id: category_info}, where all semantic
    class ids are presented and category_info record is a dict with fields
    'isthing' and 'color'
    '''
    def __init__(self, categories):
        self.taken_colors = set([0, 0, 0])
        self.categories = categories
        for category in self.categories.values():
            if category['isthing'] == 0:
                self.taken_colors.add(tuple(category['color']))

    def get_color(self, cat_id):
        def random_color(base, max_dist=30):
            new_color = base + np.random.randint(low=-max_dist,
                                                 high=max_dist+1,
                                                 size=3)
            return tuple(np.maximum(0, np.minimum(255, new_color)))

        category = self.categories[cat_id]
        if category['isthing'] == 0:
            return category['color']
        base_color_array = category['color']
        base_color = tuple(base_color_array)
        if base_color not in self.taken_colors:
            self.taken_colors.add(base_color)
            return base_color
        else:
            while True:
                color = random_color(base_color_array)
                if color not in self.taken_colors:
                    self.taken_colors.add(color)
                    return color

    def get_id(self, cat_id):
        color = self.get_color(cat_id)
        return rgb2id(color)

    def get_id_and_color(self, cat_id):
        color = self.get_color(cat_id)
        return rgb2id(color), color


def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


def id2rgb(id_map):
    if isinstance(id_map, np.ndarray):
        id_map_copy = id_map.copy()
        rgb_shape = tuple(list(id_map.shape) + [3])
        rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
        for i in range(3):
            rgb_map[..., i] = id_map_copy % 256
            id_map_copy //= 256
        return rgb_map
    color = []
    for _ in range(3):
        color.append(id_map % 256)
        id_map //= 256
    return color

    
def get_color(id):
    id = id % len(COLORMAP)
    return COLORMAP[id][0:3]

def colorize(img, colormap=None):
    indices = np.unique(img)
    indices = indices[indices != 0]

    colored_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

    for id in indices:
        mask = img == id
        if colormap is not None:
            color = colormap[id-1]
        else:
            color = get_color(id-1)
        colored_img[mask] = color

    return colored_img

def get_bbox(binary_mask):
    regions = regionprops(np.uint8(binary_mask))
    if len(regions) == 1:
        bbox = regions[0].bbox
        return bbox
    else:
        return False


def export_coco_panoptic(dataset, export_folder):
    # Create export folder
    # export_folder = os.path.join(export_folder, dataset.dataset_identifier, dataset.release['name'])
    os.makedirs(export_folder, exist_ok=True)
    
    # INFO
    info = {
        'description': dataset.release['dataset']['name'],
        'version':  dataset.release['name'],
        # 'year': '2021'
    }
    
    # CATEGORIES    
    categories = []
    for i, category in enumerate(dataset.categories):
        color = category['color'] if 'color' in category else get_color(i)
        isthing = int(category['has_instances']) if 'has_instances' in category else 0

        categories.append({
            'id': category['id'],
            'name': category['name'],
            'color': color,
            'isthing': isthing
        })
    # print(categories)

    categories_dict = {category['id']: category for category in categories}
    id_generator = IdGenerator(categories_dict)
        
    # IMAGES AND ANNOTATIONS
    images = []
    annotations = []
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]

        if sample['annotations'] is None:
            continue
        
        # Images
        image_id = i+1
        images.append({        
            'id': image_id,
            'file_name': sample['file_name'],
            'height': sample['image'].size[1],
            'width': sample['image'].size[0],
        })
        
        # Annotations
        panoptic_label = np.zeros((sample['image'].size[1], sample['image'].size[0], 3), np.uint8)
        
        segments_info = []
        for instance in sample['annotations']:
            category_id = instance['category_id']
                
            instance_id, color = id_generator.get_id_and_color(category_id)
            
            # Read the instance mask
            instance_mask = np.array(sample['segmentation_bitmap'], np.uint32) == instance['id']
            panoptic_label[instance_mask] = color
            y0, x0, y1, x1 = get_bbox(instance_mask)
            rle = mask.encode(np.array(instance_mask[:,:,None], dtype=np.uint8, order='F'))[0] # https://github.com/matterport/Mask_RCNN/issues/387#issuecomment-522671380
            area = int(mask.area(rle))
            
            info = {
                'id': instance_id,
                'category_id': category_id,
                'bbox': [x0, y0, x1-x0, y1-y0],
                'area': area,
                'iscrowd': 0
            }

            segments_info.append(info)
            
        file_name = os.path.splitext(os.path.basename(sample['name']))[0]
        label_file_name = '{}_label_{}_coco-panoptic.png'.format(file_name, dataset.labelset)
        annotations.append({
            'segments_info': segments_info,
            'file_name': label_file_name,
            'image_id': image_id,
        })        

        # # Image
        # image = sample['image']
        # export_file = os.path.join(label_export_folder, '{}.png'.format(file_name))
        # image.save(export_file)
            
        # # Instance png
        # instance_label = sample['segmentation_bitmap']
        # export_file = os.path.join(label_export_folder, '{}_gt_instance.png'.format(file_name))
        # instance_label.save(export_file)
        
        # # Colored instance png
        # instance_label_colored = colorize(np.uint8(instance_label))
        # export_file = os.path.join(label_export_folder, '{}_gt_instance_colored.png'.format(file_name))
        # Image.fromarray(img_as_ubyte(instance_label_colored)).save(export_file)
        
        # Panoptic png
        export_file = os.path.join(dataset.image_dir, label_file_name)
        Image.fromarray(panoptic_label).save(export_file)
        
        # # Semantic png
        # semantic_label = get_semantic_bitmap(instance_label, sample['annotations'])
        # export_file = os.path.join(label_export_folder, '{}_gt_semantic.png'.format(file_name))
        # Image.fromarray(img_as_ubyte(semantic_label)).save(export_file)
        
        # # Colored semantic png
        # semantic_label_colored = colorize(np.uint8(semantic_label), colormap=[c['color'] for c in categories])
        # export_file = os.path.join(label_export_folder, '{}_gt_semantic_colored.png'.format(file_name))
        # Image.fromarray(img_as_ubyte(semantic_label_colored)).save(export_file)
    
    # PUT EVERYTHING TOGETHER
    json_data = {
        'info': info,
        'categories': categories,
        'images': images,
        'annotations': annotations    
    }

    # WRITE JSON TO FILE
    file_name = os.path.join(export_folder, 'export_{}_{}.json'.format(dataset.dataset_identifier, dataset.release['name']))
    with open(file_name, 'w') as f:
        json.dump(json_data, f)

    print('Exported to {}. Images and labels in {}'.format(file_name, dataset.image_dir))
    return file_name, dataset.image_dir


def export_coco_instance(dataset, export_folder):
    # Create export folder
    # export_folder = os.path.join(export_folder, dataset.dataset_identifier, dataset.release['name'])
    os.makedirs(export_folder, exist_ok=True)

    info = {
        'description': dataset.release['dataset']['name'],
        # 'url': 'https://segments.ai/test/test',
        'version':  dataset.release['name'],
        # 'year': 2020,
        # 'contributor': 'Segments.ai',
    }

    # licenses = [{
    #     'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/',
    #     'id': 1,
    #     'name': 'Attribution-NonCommercial-ShareAlike License'
    # }]

    categories = dataset.categories
    task_type = dataset.task_type
    # for i, category in enumerate(dataset.project_info['label_taxonomy']):
    #     categories.append({
    #         'id': i+1,
    #         'supercategory': 'object',
    #         'name': category
    #     })

    images = []
    annotations = []

    annotation_id = 1
    for i in tqdm(range(len(dataset))):        
        sample = dataset[i]

        if sample['annotations'] is None:
            continue
        
        image_id = i+1
        images.append({        
            'id': image_id,
            # 'license': 1,
            'file_name': sample['file_name'],
            'height': sample['image'].size[1],
            'width': sample['image'].size[0],
    #         'date_captured': "2013-11-14 17:02:52",
    #         'coco_url': "http://images.cocodataset.org/val2017/000000397133.jpg",
    #         'flickr_url': "http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg",        
        })
        
        for instance in sample['annotations']:
            category_id = instance['category_id']

            annotation = {
                'id': annotation_id,
                'image_id': image_id,
                'category_id': category_id,
            }

            # Segmentation bitmap
            if task_type == 'segmentation-bitmap':                
                instance_mask = np.array(sample['segmentation_bitmap'], np.uint32) == instance['id']
                bbox = get_bbox(instance_mask)
                if not bbox:
                    continue
                    
                y0, x0, y1, x1 = bbox
                # rle = mask.encode(np.asfortranarray(instance_mask))
                rle = mask.encode(np.array(instance_mask[:,:,None], dtype=np.uint8, order='F'))[0] # https://github.com/matterport/Mask_RCNN/issues/387#issuecomment-522671380
        #         instance_mask_crop = instance_mask[y0:y1, x0:x1]
        #         rle = mask.encode(np.asfortranarray(instance_mask_crop))
        #         plt.imshow(instance_mask_crop)
        #         plt.show()
                
                area = int(mask.area(rle))
                rle['counts'] = rle['counts'].decode('ascii')

                annotation.update({
                    'bbox': [x0, y0, x1-x0, y1-y0],
        #             'bbox_mode': BoxMode.XYWH_ABS,
                    'segmentation': rle,
                    'area': area,
                    'iscrowd': 0,
                })

            # Bounding boxes
            elif task_type == 'bboxes':
                points = instance['points']
                x0 = points[0][0]
                y0 = points[0][1]
                x1 = points[1][0]
                y1 = points[1][1]

                annotation.update({
                    'bbox': [x0, y0, x1-x0, y1-y0],
                })

            else:
                assert False

            annotations.append(annotation)
            annotation_id += 1
            
    json_data = {
        'info': info,
        # 'licenses': licenses,
        'categories': categories,
        'images': images,
        'annotations': annotations    
    #     'segment_info': [] # Only in Panoptic annotations
    }

    file_name = os.path.join(export_folder, 'export_{}_{}.json'.format(dataset.dataset_identifier, dataset.release['name']))
    with open(file_name, 'w') as f:
        json.dump(json_data, f)

    print('Exported to {}. Images and labels in {}'.format(file_name, dataset.image_dir))
    return file_name, dataset.image_dir


def export_yolo(dataset, export_folder):
    if dataset.task_type != 'bboxes':
        print('You can only export bounding box datasets to YOLO format.')
        return

    for i in tqdm(range(len(dataset))):        
        sample = dataset[i]

        if 'annotations' in sample and len(sample['annotations']) > 0:
            image_name = os.path.splitext(os.path.basename(sample['name']))[0]
            image_width = sample['image'].width
            image_height = sample['image'].height
            
            file_name = '{}/{}.txt'.format(dataset.image_dir, image_name)
    #         print(file_name)       
            
            with open(file_name, 'w') as f:
                for annotation in sample['annotations']:
                    category_id = annotation['category_id']
                    [[x0, y0], [x1, y1]] = annotation['points']

                    # Normalize
                    x0, x1 = x0 / image_width, x1 / image_width
                    y0, y1 = y0 / image_height, y1 / image_height

                    # Get center, width and height of bbox
                    x_center = (x0 + x1) / 2
                    y_center = (y0 + y1) / 2
                    width = abs(x1 - x0)
                    height = abs(y1 - y0)

                    # Save it to the file
    #                 print(category_id, x_center, y_center, width, height)
                    f.write('{} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(category_id, x_center, y_center, width, height))

    print('Exported. Images and labels in {}'.format(dataset.image_dir))
    return dataset.image_dir

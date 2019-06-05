import os
import numpy as np

from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

from mrcnn import utils


class HouseDataset(utils.Dataset):

    def __init__(self):
        super(self.__class__, self).__init__()

        self.load_small = False
        self.coco = None
        self.image_dir = ''

    def load_dataset(self, path):
        annotation_path = path
        image_dir = os.path.join(os.path.dirname(path), "images")

        assert os.path.exists(annotation_path) and os.path.exists(image_dir)

        self.coco = COCO(annotation_path)
        self.image_dir = image_dir

        # Load all classes (Only Building in this version)
        class_ids = self.coco.getCatIds()

        # Load all images
        image_ids = list(self.coco.imgs.keys())

        # register classes
        for class_id in class_ids:
            self.add_class("house-mapping", class_id, self.coco.loadCats(class_id)[0]["name"])

        # Register Images
        for image_id in image_ids:
            assert(os.path.exists(os.path.join(image_dir, self.coco.imgs[image_id]['file_name'])))
            self.add_image(
                "house-mapping", image_id=image_id,
                path=os.path.join(image_dir, self.coco.imgs[image_id]['file_name']),
                width=self.coco.imgs[image_id]["width"],
                height=self.coco.imgs[image_id]["height"],
                annotations=self.coco.loadAnns(self.coco.getAnnIds(imgIds=[image_id], catIds=class_ids, iscrowd=None)))

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        assert image_info["source"] == "house-mapping"

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "house-mapping.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"], image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue

                # Ignore the notion of "is_crowd" as specified in the coco format
                # as we donot have the said annotation in the current version of the dataset

                instance_masks.append(m)
                class_ids.append(class_id)
        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(self.__class__, self).load_mask(image_id)

    def image_reference(self, image_id):
        return "house-mapping::{}".format(image_id)

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

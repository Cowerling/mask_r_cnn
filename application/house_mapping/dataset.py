import os
import numpy as np
import rasterio
from rasterio.windows import Window
import math

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


class GDALDataset(object):

    def __init__(self, path):
        self.data_source = rasterio.open(path)
        self.band_count = self.data_source.count
        self.row_count = self.data_source.height
        self.column_count = self.data_source.width
        self.epsg = self.data_source.crs.to_epsg()

    def generate_bounds(self, size, buffer, batch_size, extent=None):
        size_row = size[0]
        size_column = size[1]
        buffer_row = math.floor(buffer[0] / math.fabs(self.data_source.transform[4]))
        buffer_column = math.floor(buffer[1] / self.data_source.transform[0])

        assert buffer_row < size_row
        assert buffer_column < size_column

        if extent is None:
            row_min = 0
            column_min = 0
            row_max = self.row_count
            column_max = self.column_count
        else:
            x_min = extent[0]
            y_min = extent[1]
            x_max = extent[2]
            y_max = extent[3]

            row_min, column_min = self.data_source.index(x_min, y_max)
            row_max, column_max = self.data_source.index(x_max, y_min)

            if row_min < 0:
                row_min = 0

            if column_min < 0:
                column_min = 0

            if row_max > self.row_count:
                row_max = self.row_count

            if column_max > self.column_count:
                column_max = self.column_count

        all_bounds = []
        bounds = []

        for row in range(row_min, row_max, size_row - buffer_row):
            for column in range(column_min, column_max, size_column - buffer_column):
                _size_row = size_row if row + size_row <= self.row_count else self.row_count - row
                _size_column = size_column if column + size_column <= self.column_count else self.column_count - column

                if len(bounds) < batch_size:
                    bounds.append((row, column, _size_row, _size_column))

                if len(bounds) == batch_size:
                    all_bounds.append(bounds)
                    bounds = []

        if len(bounds) != 0:
            all_bounds.append(bounds)

        return all_bounds, (size_row, size_column, buffer_row, buffer_column)

    def get_data(self, bound):
        row = bound[0]
        column = bound[1]
        size_row = bound[2]
        size_column = bound[3]

        data = self.data_source.read(window=Window(column, row, size_column, size_row))
        data = data.transpose(1, 2, 0)

        return data

    def get_transform(self, row, column):
        x_top_left, y_top_left = self.data_source.xy(row, column)
        x_top_left -= 0.15 / 2
        y_top_left += 0.15 / 2

        return rasterio.transform.from_origin(x_top_left,
                                              y_top_left,
                                              math.fabs(self.data_source.transform[0]),
                                              math.fabs(self.data_source.transform[4]))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.data_source.close()

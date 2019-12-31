import rasterio
from rasterio.windows import Window
import math
import psycopg2
import ogr
import json
from rasterio import features
import numpy as np
import tqdm

from mrcnn import utils


class GDALDataset(object):

    def __init__(self, path):
        self.data_source = rasterio.open(path)
        self.band_count = self.data_source.count
        self.row_count = self.data_source.height
        self.column_count = self.data_source.width
        self.epsg = self.data_source.crs.to_epsg()
        self.METER_TO_DEGREE = 0.0000094

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

            row_min = 0 if row_min < 0 else row_min
            column_min = 0 if column_min < 0 else column_min
            row_max = self.row_count - 1 if row_max >= self.row_count else row_max
            column_max = self.column_count - 1 if column_max >= self.column_count else column_max

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
        x_top_left -= math.fabs(self.data_source.transform[0]) / 2
        y_top_left += math.fabs(self.data_source.transform[4]) / 2

        return rasterio.transform.from_origin(x_top_left,
                                              y_top_left,
                                              math.fabs(self.data_source.transform[0]),
                                              math.fabs(self.data_source.transform[4]))

    def revise_buffer(self, buffer):
        if self.data_source.crs.is_geographic:
            return buffer[0] * self.METER_TO_DEGREE, buffer[1] * self.METER_TO_DEGREE
        else:
            return buffer

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.data_source.close()


class SpatialDataset(utils.Dataset):

    def __init__(self, source, class_names):
        super(self.__class__, self).__init__()

        self.data_references = []

        self.source = source

        for class_name in class_names:
            self.add_class(source, len(self.class_info), class_name)

    def load_data(self, rs_image_path, database, user, password, host, port, mask_table, bound_table, condition):
        connection_string = 'dbname={} user={} password={} host={} port={}'.format(database,
                                                                                   user,
                                                                                   password,
                                                                                   host,
                                                                                   port)

        self.data_references.append({
            'path': rs_image_path,
            'connection_string': connection_string,
            'mask_table': mask_table
        })

        with GDALDataset(rs_image_path) as dataset:
            with psycopg2.connect(connection_string) as connection:
                with connection.cursor() as cursor:
                    cursor.execute('SELECT id, ST_AsText(geom) FROM ' + bound_table +
                                   ('' if condition is None else ' WHERE ' + condition))
                    bounds = cursor.fetchall()

                    for bound in tqdm.tqdm(bounds):
                        bound_id = bound[0]
                        bound_envelop = ogr.CreateGeometryFromWkt(bound[1]).GetEnvelope()

                        row_min, column_min = dataset.data_source.index(bound_envelop[0], bound_envelop[3])
                        row_max, column_max = dataset.data_source.index(bound_envelop[1], bound_envelop[2])

                        row_min = 0 if row_min < 0 else row_min
                        column_min = 0 if column_min < 0 else column_min
                        row_max = dataset.row_count - 1 if row_max >= dataset.row_count else row_max
                        column_max = dataset.column_count - 1 if column_max >= dataset.column_count else column_max

                        row = row_min
                        column = column_min
                        size_row = row_max - row_min + 1
                        size_column = column_max - column_min + 1

                        transform = dataset.get_transform(row, column)

                        self.add_image(self.source,
                                       image_id='{}-{}-{}-{}'.format(host, database, bound_table, bound_id),
                                       path=None,
                                       reference_id=len(self.data_references) - 1,
                                       bound=(row, column, size_row, size_column),
                                       transform=transform)

    def load_image(self, image_id):
        image_info = self.image_info[image_id]

        reference_id = image_info['reference_id']
        bound = image_info['bound']

        rs_image_path = self.data_references[reference_id]['path']

        with GDALDataset(rs_image_path) as dataset:
            image = dataset.get_data(bound)

        return image

    def image_reference(self, image_id):
        image_info = self.image_info[image_id]
        reference_id = image_info['reference_id']

        return '{}.{}.{}'.format(image_id, self.data_references[reference_id]['path'], image_info['bound'])

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]

        inner_id = image_info['id']
        bound = image_info['bound']
        transform = image_info['transform']

        reference_id = image_info['reference_id']
        data_reference = self.data_references[reference_id]
        connection_string = data_reference['connection_string']
        mask_table = data_reference['mask_table']

        bound_id = inner_id.split('-')[-1]

        masks = []
        mask_classes = []

        with psycopg2.connect(connection_string) as connection:
            with connection.cursor() as cursor:
                cursor.execute('SELECT ST_AsText(geom), class FROM {} WHERE bound = {}'.format(mask_table, bound_id))
                spatial_masks = cursor.fetchall()

                for spatial_mask_with_class in spatial_masks:
                    spatial_mask = ogr.CreateGeometryFromWkt(spatial_mask_with_class[0])
                    mask = features.geometry_mask(geometries=[json.loads(spatial_mask.ExportToJson())],
                                                  out_shape=(bound[2], bound[3]),
                                                  transform=transform,
                                                  all_touched=True,
                                                  invert=True)

                    masks.append(mask)

                    mask_class = spatial_mask_with_class[1]
                    mask_classes.append(mask_class)

        masks = np.array(masks)
        masks = masks.transpose(1, 2, 0)

        class_ids = np.array([self.class_names.index(mask_class) for mask_class in mask_classes])
        class_ids = class_ids.astype(np.int32)

        return masks, class_ids

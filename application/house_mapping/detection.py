import os
import tqdm
import numpy as np
import psycopg2
from postgis.psycopg import register
import ogr
from rasterio import features

from mrcnn import config
from mrcnn import models

from application.house_mapping.dataset import GDALDataset


def to_wkt(postgis_geometry):
    postgis_geometry_wkt = postgis_geometry.wkt_coords
    postgis_geometry_wkt = postgis_geometry_wkt.replace('(', '')
    postgis_geometry_wkt = postgis_geometry_wkt.replace(')', '')
    postgis_geometry_wkt = 'POLYGON ((' + postgis_geometry_wkt + '))'
    return postgis_geometry_wkt


def get_mask_box(block_id, mask_box_list):
    cursor.execute('SELECT ' +
                   mask_table + '.geom AS mask, ' +
                   box_table + '.geom AS box, ' +
                   mask_table + '.id AS id '
                               'FROM ' + mask_table + ' LEFT JOIN ' + box_table +
                   ' ON ' + mask_table + '.id = ' + box_table + '.id WHERE block_id = %s', [block_id])
    _mask_box_list = cursor.fetchall()

    for mask_box in _mask_box_list:
        mask_box_list.append((ogr.CreateGeometryFromWkt(to_wkt(mask_box[0])),
                              ogr.CreateGeometryFromWkt(to_wkt(mask_box[1])),
                              mask_box[2]))


def envelope_to_box(envelope):
    box_wkt = 'POLYGON (({} {}, {} {}, {} {}, {} {}, {} {}))'.format(envelope[0],
                                                                     envelope[2],
                                                                     envelope[1],
                                                                     envelope[2],
                                                                     envelope[1],
                                                                     envelope[3],
                                                                     envelope[0],
                                                                     envelope[3],
                                                                     envelope[0],
                                                                     envelope[2])
    return ogr.CreateGeometryFromWkt(box_wkt)


class InferenceConfig(config.Config):
    NAME = 'house-mapping'

    IMAGES_PER_GPU = 5
    GPU_COUNT = 1

    NUM_CLASSES = 1 + 1

    IMAGE_MAX_DIM = 1024
    IMAGE_MIN_DIM = 1024


config = InferenceConfig()
class_names = ['BG', 'house']

work_space = '/home/cowerling/house_mapping'

database = 'house_mapping'
user = 'postgres'
password = 'Admin948'
host = '116.62.21.158'
port = '5432'

mask_table = 'mask'
box_table = 'box'

box_intersection_ratio = 0.3

model = models.MaskRCNN(mode='inference', model_dir=os.path.join(work_space, 'log'), config=config)
model.load_weights(os.path.join(work_space, 'weight/mask_rcnn_house-mapping_0160.h5'), by_name=True)

with GDALDataset(os.path.join(work_space, 'image/honghu.tif')) as dataset:
    bounds = dataset.generate_bounds(size=(1024, 1024),
                                     buffer=(10, 10),
                                     batch_size=config.BATCH_SIZE)
    print('bounds count: {}'.format(len(bounds)))

    connection = psycopg2.connect(database=database,
                                  user=user,
                                  password=password,
                                  host=host,
                                  port=port)
    register(connection)
    cursor = connection.cursor()

    count = 0

    for batch_bounds in tqdm.tqdm(bounds):
        batch_bounds_size = len(batch_bounds)
        print('batch bounds size: {}'.format(batch_bounds_size))

        if batch_bounds_size < config.BATCH_SIZE:
            batch_bounds = np.concatenate((batch_bounds, batch_bounds[0: 1] *
                                           (config.BATCH_SIZE - batch_bounds_size)), axis=0)

        results = model.detect([dataset.get_data(bound) for bound in batch_bounds], verbose=0)

        for index, result in enumerate(results):
            if index >= batch_bounds_size:
                break

            bound = batch_bounds[index]
            row = bound[0]
            column = bound[1]

            pre_mask_box_list = []
            get_mask_box('{}-{}'.format(row - 1, column - 1), pre_mask_box_list)
            get_mask_box('{}-{}'.format(row, column - 1), pre_mask_box_list)
            get_mask_box('{}-{}'.format(row - 1, column), pre_mask_box_list)

            transform = dataset.get_transform(row, column)

            for i, class_id in enumerate(result['class_ids']):
                box = result['rois'][i]
                class_name = class_names[class_id]
                score = result['scores'][i]
                mask = result['masks'][:, :, i].astype(np.int16)

                spatial_box_top_left = transform * (box[1], box[0])
                spatial_box_bottom_right = transform * (box[3], box[2])
                spatial_box = envelope_to_box((spatial_box_top_left[0],
                                               spatial_box_bottom_right[0],
                                               spatial_box_bottom_right[1],
                                               spatial_box_top_left[1]))

                spatial_mask_shapes = features.shapes(mask, mask=(mask != 0), transform=transform)
                spatial_mask_shape = next(spatial_mask_shapes)[0]
                spatial_mask_coordinate_sizes = [len(coordinates) for coordinates in spatial_mask_shape['coordinates']]
                spatial_mask_coordinates = spatial_mask_shape['coordinates'][
                    spatial_mask_coordinate_sizes.index(max(spatial_mask_coordinate_sizes))]

                spatial_mask_ring = ogr.Geometry(ogr.wkbLinearRing)
                for spatial_mask_coordinate in spatial_mask_coordinates:
                    spatial_mask_ring.AddPoint(spatial_mask_coordinate[0], spatial_mask_coordinate[1])
                spatial_mask = ogr.Geometry(ogr.wkbPolygon)
                spatial_mask.AddGeometry(spatial_mask_ring)
                spatial_mask.FlattenTo2D()

                for pre_mask_box in pre_mask_box_list:
                    pre_mask = pre_mask_box[0]
                    pre_box = pre_mask_box[1]
                    pre_id = pre_mask_box[2]

                    box_union_area = pre_box.Union(spatial_box).GetArea()
                    box_intersection_area = pre_box.Intersection(spatial_box).GetArea()

                    if box_intersection_area / box_union_area < box_intersection_ratio:
                        continue

                    spatial_mask = pre_mask.Union(spatial_mask)
                    spatial_box = envelope_to_box(spatial_mask.GetEnvelope())

                    cursor.execute('DELETE FROM ' + mask_table + ' WHERE id = %s', pre_id)
                    cursor.execute('DELETE FROM ' + box_table + ' WHERE id = %s', pre_id)
                    print('delete mask and box at {}'.format(pre_id))

                count += 1

                cursor.execute('INSERT INTO ' + mask_table + ' VALUES (%s, ST_GeomFromText(%s, %s), %s, %s)',
                               [count, spatial_mask.ExportToWkt(), dataset.epsg, class_name, score.item()])
                cursor.execute('INSERT INTO ' + box_table + ' VALUES (%s, ST_GeomFromText(%s, %s), %s)',
                               [count, spatial_box.ExportToWkt(), dataset.epsg, '{}-{}'.format(row, column)])
                print('insert mask and box at {}'.format(count))

                connection.commit()

            print('bound row: {}, column: {} finished'.format(row, column))
            print('----------------------------------')

    cursor.close()
    connection.close()

print('mission completed')

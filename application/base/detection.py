import os
import tqdm
import numpy as np
import psycopg2
import ogr
from rasterio import features
import json
from termcolor import colored
import matplotlib.pyplot as plt

from mrcnn import config
from mrcnn import models

from application.base.dataset import GDALDataset


def get_masks(block_id, mask_list, cursor, mask_table):
    cursor.execute('SELECT '
                   'ST_AsText(mask), ST_AsText(box), id '
                   'FROM ' + mask_table + ' WHERE block = %s', [block_id])
    _mask_list = cursor.fetchall()

    for _mask in _mask_list:
        single_mask = ogr.CreateGeometryFromWkt(_mask[0])
        single_box = ogr.CreateGeometryFromWkt(_mask[1])
        single_id = _mask[2]

        single_mask = ensure_geometry(single_mask)

        if single_mask.IsValid():
            mask_list.append((single_mask, single_box, single_id))
        else:
            print(colored('WARNING: previous mask at {} is not valid, not append to list'.format(single_id), 'red'))


def envelope_to_box(envelope):
    box_wkt = 'MULTIPOLYGON ((({} {}, {} {}, {} {}, {} {}, {} {})))'.format(envelope[0],
                                                                            envelope[2],
                                                                            envelope[0],
                                                                            envelope[3],
                                                                            envelope[1],
                                                                            envelope[3],
                                                                            envelope[1],
                                                                            envelope[2],
                                                                            envelope[0],
                                                                            envelope[2])
    return ogr.CreateGeometryFromWkt(box_wkt)


def ensure_geometry(geometry):
    if not geometry.IsValid():
        print(colored('use buffer to fix geometry', 'blue'))
        geometry = geometry.Buffer(0)

    return geometry


def detect(rs_image_path, class_names,
           weight_path, log_dir,
           database, user, password, host, port, mask_table, block_table,
           bound_size, bound_buffer,
           reset=True, images_per_gpu=5, gpu_count=1,
           mean_pixel_values=None,
           show_mask=False):

    class InferenceConfig(config.Config):
        NAME = os.path.splitext(os.path.basename(rs_image_path))[0]

        IMAGES_PER_GPU = images_per_gpu
        GPU_COUNT = gpu_count

        NUM_CLASSES = len(class_names)

        IMAGE_MAX_DIM = bound_size[0]
        IMAGE_MIN_DIM = bound_size[1]

        IMAGE_CHANNEL_COUNT = config.Config.IMAGE_CHANNEL_COUNT if mean_pixel_values is None else len(mean_pixel_values)
        MEAN_PIXEL = config.Config.MEAN_PIXEL if mean_pixel_values is None else np.array(mean_pixel_values)

    inference_config = InferenceConfig()

    model = models.MaskRCNN(mode='inference', model_dir=log_dir, config=inference_config)
    model.load_weights(weight_path, by_name=True)

    with GDALDataset(rs_image_path) as dataset:
        bounds, bound_size_buffer = dataset.generate_bounds(size=bound_size,
                                                            buffer=bound_buffer,
                                                            batch_size=inference_config.BATCH_SIZE)
        bound_offset_row = bound_size_buffer[0] - bound_size_buffer[2]
        bound_offset_column = bound_size_buffer[1] - bound_size_buffer[3]
        print('bounds count: {}, size: ({} {}), buffer: ({} {})'.format(len(bounds),
                                                                        bound_size_buffer[0],
                                                                        bound_size_buffer[1],
                                                                        bound_size_buffer[2],
                                                                        bound_size_buffer[3]))

        connection = psycopg2.connect(database=database,
                                      user=user,
                                      password=password,
                                      host=host,
                                      port=port)
        cursor = connection.cursor()

        # create mask and block table in database
        print('create mask table: {}'.format(mask_table))
        cursor.execute('CREATE TABLE IF NOT EXISTS {} '
                       '('
                       'id bigint NOT NULL PRIMARY KEY, '
                       'mask geometry(MultiPolygon, {}), '
                       'class varchar(30), '
                       'score numeric, '
                       'box geometry(MultiPolygon, {}), '
                       'block varchar(50), '
                       'expand integer'
                       ')'.format(mask_table, dataset.epsg, dataset.epsg))
        cursor.execute('CREATE INDEX IF NOT EXISTS sidx_{}_mask ON {} USING GIST (mask)'.
                       format(mask_table, mask_table))

        print('create block table: {}'.format(block_table))
        cursor.execute('CREATE TABLE IF NOT EXISTS {} '
                       '('
                       'id varchar(50) NOT NULL PRIMARY KEY, '
                       'geom geometry(MultiPolygon, {})'
                       ')'.format(block_table, dataset.epsg))
        cursor.execute('CREATE INDEX IF NOT EXISTS sidx_{}_geom ON {} USING GIST (geom)'.
                       format(block_table, block_table))

        connection.commit()

        if reset:
            print('clear table: {}, {}'.format(mask_table, block_table))
            cursor.execute('DELETE FROM ' + mask_table)
            cursor.execute('DELETE FROM ' + block_table)
            connection.commit()

        count = 0

        for batch_bounds in tqdm.tqdm(bounds):
            batch_bounds_size = len(batch_bounds)
            print('batch bounds size: {}'.format(batch_bounds_size))

            if batch_bounds_size < inference_config.BATCH_SIZE:
                batch_bounds = np.concatenate((batch_bounds, batch_bounds[0: 1] *
                                               (inference_config.BATCH_SIZE - batch_bounds_size)), axis=0)

            results = model.detect([dataset.get_data(bound) for bound in batch_bounds], verbose=0)

            for index, result in enumerate(results):
                if index >= batch_bounds_size:
                    break

                bound = batch_bounds[index]
                row = bound[0]
                column = bound[1]

                pre_masks = []
                get_masks('{}-{}'.format(row - bound_offset_row, column - bound_offset_column),
                          pre_masks,
                          cursor, mask_table)
                get_masks('{}-{}'.format(row, column - bound_offset_column),
                          pre_masks,
                          cursor,
                          mask_table)
                get_masks('{}-{}'.format(row - bound_offset_row, column),
                          pre_masks,
                          cursor,
                          mask_table)
                get_masks('{}-{}'.format(row - bound_offset_row, column + bound_offset_column),
                          pre_masks,
                          cursor,
                          mask_table)
                print('find {} previous mask and box'.format(len(pre_masks)))

                transform = dataset.get_transform(row, column)

                for i, class_id in enumerate(result['class_ids']):
                    box = result['rois'][i]
                    class_name = class_names[class_id]
                    score = result['scores'][i]
                    mask = result['masks'][:, :, i].astype(np.int16)

                    if show_mask:
                        plt.matshow(mask)
                        plt.show()

                    spatial_box_top_left = transform * (box[1], box[0])
                    spatial_box_bottom_right = transform * (box[3], box[2])
                    spatial_box = envelope_to_box((spatial_box_top_left[0],
                                                   spatial_box_bottom_right[0],
                                                   spatial_box_bottom_right[1],
                                                   spatial_box_top_left[1]))

                    spatial_mask_shapes = features.shapes(mask, mask=(mask != 0), transform=transform)

                    spatial_mask = ogr.Geometry(ogr.wkbMultiPolygon)

                    try:
                        while True:
                            sub_spatial_mask_shape = next(spatial_mask_shapes)[0]
                            sub_spatial_mask = ogr.CreateGeometryFromJson(json.dumps(sub_spatial_mask_shape))
                            spatial_mask.AddGeometry(sub_spatial_mask)

                    except StopIteration:
                        pass

                    spatial_mask = ensure_geometry(spatial_mask)

                    mask_expand_count = 0

                    if spatial_mask.IsValid():
                        remove_pre_mask_indices = []

                        for pre_mask_index, _pre_mask in enumerate(pre_masks):
                            pre_mask = _pre_mask[0]
                            pre_box = _pre_mask[1]
                            pre_id = _pre_mask[2]

                            mask_intersection = pre_mask.Intersection(spatial_mask)

                            if mask_intersection is None:
                                continue

                            if not mask_intersection.IsValid() or mask_intersection.GetArea() == 0:
                                continue

                            spatial_mask = pre_mask.Union(spatial_mask)
                            spatial_mask = ensure_geometry(spatial_mask)

                            spatial_box = pre_box.Union(spatial_box)
                            spatial_box = ensure_geometry(spatial_box)

                            mask_expand_count = mask_expand_count + 1

                            cursor.execute('DELETE FROM ' + mask_table + ' WHERE id = %s', [pre_id])
                            connection.commit()
                            print(colored('delete previous mask and box at {}'.format(pre_id), 'yellow'))

                            remove_pre_mask_indices.append(pre_mask_index)

                        pre_mask_indices = [inner_i for inner_i in range(len(pre_masks))]
                        retain_pre_mask_indices = list(set(pre_mask_indices) - set(remove_pre_mask_indices))
                        pre_masks = [pre_masks[inner_i] for inner_i in retain_pre_mask_indices]

                    count = count + 1

                    if spatial_mask.GetGeometryType() == ogr.wkbPolygon:
                        spatial_mask = ogr.ForceToMultiPolygon(spatial_mask)

                    if spatial_box.GetGeometryType() == ogr.wkbPolygon:
                        spatial_box = ogr.ForceToMultiPolygon(spatial_box)

                    if not spatial_mask.IsValid():
                        print(colored('WARNING: mask at {} is not valid'.format(count), 'red'))

                    cursor.execute('INSERT INTO ' + mask_table +
                                   ' (id, mask, class, score, box, block, expand) VALUES '
                                   '(%s, ST_GeomFromText(%s, %s), %s, %s, ST_GeomFromText(%s, %s), %s, %s)',
                                   [count,
                                    spatial_mask.ExportToWkt(),
                                    dataset.epsg,
                                    class_name,
                                    score.item(),
                                    spatial_box.ExportToWkt(),
                                    dataset.epsg,
                                    '{}-{}'.format(row, column),
                                    mask_expand_count])
                    connection.commit()
                    print(colored('insert mask and box at {}'.format(count), 'green'))

                    pre_masks.append((spatial_mask, spatial_box, count))

                spatial_bound_top_left = transform * (0, 0)
                spatial_bound_bottom_right = transform * (bound[3] - 1, bound[2] - 1)
                spatial_bound = envelope_to_box((spatial_bound_top_left[0],
                                                 spatial_bound_bottom_right[0],
                                                 spatial_bound_bottom_right[1],
                                                 spatial_bound_top_left[1]))

                cursor.execute('INSERT INTO ' + block_table +
                               ' (id, geom) VALUES '
                               '(%s, ST_GeomFromText(%s, %s))',
                               ['{}-{}'.format(row, column),
                                spatial_bound.ExportToWkt(),
                                dataset.epsg])

                connection.commit()

                print('bound index: {}, row: {}, column: {} finished'.format(index, row, column))
                print('----------------------------------')

        cursor.close()
        connection.close()

    print('mission completed')

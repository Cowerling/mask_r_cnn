import os

from application.base.training import train
from application.base.training import TrainingMode

from application.base.dataset import SpatialDataset
import numpy as np
from mrcnn import visualize


work_space = '/home/cowerling/house_mapping'

# dataset = SpatialDataset()
# dataset.load_data(source='house',
#                   rs_image_path=os.path.join(work_space, 'image/sample_val.tif'),
#                   database='house_mapping',
#                   user='postgres',
#                   password='Admin948',
#                   host='10.6.72.213',
#                   port=5432,
#                   mask_table='sample_mask_val',
#                   bound_table='sample_bound_val')
# dataset.prepare()
#
# image_ids = np.random.choice(dataset.image_ids, 4)
# for image_id in dataset.image_ids:
#     image = dataset.load_image(image_id)
#     mask, class_ids = dataset.load_mask(image_id)
#     visualize.display_top_masks(image, mask, class_ids, dataset.class_names)

train_sources = [{
    'rs_image_path': os.path.join(work_space, 'image/sample.tif'),
    'database': 'house_mapping',
    'user': 'postgres',
    'password': 'Admin948',
    'host': '10.6.72.213',
    'port': 5432,
    'mask_table': 'sample_mask',
    'bound_table': 'sample_bound',
    'condition': None
}]

validation_sources = [{
    'rs_image_path': os.path.join(work_space, 'image/sample_val.tif'),
    'database': 'house_mapping',
    'user': 'postgres',
    'password': 'Admin948',
    'host': '10.6.72.213',
    'port': 5432,
    'mask_table': 'sample_mask_val',
    'bound_table': 'sample_bound_val',
    'condition': None
}]

train(source='house', class_names=['house'],
      train_sources=train_sources,
      validation_sources=validation_sources,
      weight_path=os.path.join(work_space, 'weight/pretrained_weights.h5'),
      log_dir=os.path.join(work_space, 'log'),
      epochs=10,
      training_mode=TrainingMode.HEADS)

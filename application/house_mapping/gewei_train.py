import os

from application.base.training import train
from application.base.training import TrainingMode
from application.base.training import augment

work_space = '/home/cowerling/house_mapping'

train_sources = [{
    'rs_image_path': os.path.join(work_space, 'image/gewei.tif'),
    'database': 'house_mapping',
    'user': 'postgres',
    'password': 'Admin948',
    'host': '10.6.72.213',
    'port': 5432,
    'mask_table': 'gewei_mask',
    'bound_table': 'gewei_bound',
    'condition': 'id <= 2000'
}]

validation_sources = [{
    'rs_image_path': os.path.join(work_space, 'image/gewei.tif'),
    'database': 'house_mapping',
    'user': 'postgres',
    'password': 'Admin948',
    'host': '10.6.72.213',
    'port': 5432,
    'mask_table': 'gewei_mask',
    'bound_table': 'gewei_bound',
    'condition': 'id > 2000'
}]

train(source='gewei_house', class_names=['house'],
      train_sources=train_sources,
      validation_sources=validation_sources,
      weight_path=os.path.join(work_space, 'weight/pretrained_weights.h5'),
      log_dir=os.path.join(work_space, 'log'),
      image_dimension=(512, 512),
      epochs=100,
      training_mode=TrainingMode.HEADS,
      augmentation=augment(scale=(0.5, 1.5), rotate=(-45, 45), fliplr=0.5),
      name='gewei')

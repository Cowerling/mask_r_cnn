import os

from application.base.training import train
from application.base.training import TrainingMode

work_space = '/home/cowerling/house_mapping'

train_sources = [{
    'rs_image_path': os.path.join(work_space, 'image/gewei_dsm_normal_83_3.tif'),
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
    'rs_image_path': os.path.join(work_space, 'image/gewei_dsm_normal_83_3.tif'),
    'database': 'house_mapping',
    'user': 'postgres',
    'password': 'Admin948',
    'host': '10.6.72.213',
    'port': 5432,
    'mask_table': 'gewei_mask',
    'bound_table': 'gewei_bound',
    'condition': 'id > 2000'
}]

train(source='gewei_dsm_house', class_names=['house'],
      train_sources=train_sources,
      validation_sources=validation_sources,
      weight_path=os.path.join(work_space, 'weight/pretrained_weights.h5'),
      log_dir=os.path.join(work_space, 'log'),
      image_dimension=(512, 512),
      steps_per_epoch=400, validation_steps=200,
      epochs=100,
      training_mode=TrainingMode.HEADS,
      name='gewei_dsm')

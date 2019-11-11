import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))

from application.base.training import train
from application.base.training import TrainingMode
from application.base.training import augment

work_space = '/home/cowerling/DeepLearning/house/gewei'

train_sources = [{
    'rs_image_path': os.path.join(work_space, 'image/gewei.tif'),
    'database': 'house',
    'user': 'postgres',
    'password': 'Cowerling',
    'host': 'localhost',
    'port': 5432,
    'mask_table': 'sample_mask_gewei',
    'bound_table': 'sample_bound_gewei',
    'condition': 'id <= 2000'
}]

validation_sources = [{
    'rs_image_path': os.path.join(work_space, 'image/gewei.tif'),
    'database': 'house',
    'user': 'postgres',
    'password': 'Cowerling',
    'host': 'localhost',
    'port': 5432,
    'mask_table': 'sample_mask_gewei',
    'bound_table': 'sample_bound_gewei',
    'condition': 'id > 2000'
}]

train(source='house_gewei', class_names=['house'],
      train_sources=train_sources,
      validation_sources=validation_sources,
      weight_path=os.path.join(work_space, 'weight/pretrained_weights.h5'),
      log_dir=os.path.join(work_space, 'log'),
      image_dimension=(512, 512),
      epochs=1000,
      steps_per_epoch=100, validation_steps=50,
      training_mode=TrainingMode.HEADS,
      gpu_count=2,
      augmentation=augment(scale=(0.5, 1.5), rotate=(-45, 45), fliplr=0.5),
      name='house_gewei')

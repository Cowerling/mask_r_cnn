from mrcnn import config
from mrcnn import models

from application.house_mapping.dataset import HouseDataset


class TrainConfig(config.Config):
    NAME = 'house-mapping'

    IMAGES_PER_GPU = 5
    GPU_COUNT = 1

    NUM_CLASSES = 1 + 1

    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 50

    IMAGE_MAX_DIM = 320
    IMAGE_MIN_DIM = 320

    DETECTION_MIN_CONFIDENCE = 0.9


config = TrainConfig()

model = models.MaskRCNN(mode='training', config=config, model_dir='/home/cowerling/house_mapping/log')
model.load_weights('/home/cowerling/house_mapping/weight/mask_rcnn_house-mapping_0036.h5', by_name=True)

dataset_train = HouseDataset()
dataset_train.load_dataset('./image/train/annotation-small.json')
dataset_train.prepare()

dataset_validation = HouseDataset()
dataset_validation.load_dataset('./image/val/annotation-small.json')
dataset_validation.prepare()

# print('Training heads')
# model.train(dataset_train, dataset_validation, learning_rate=config.LEARNING_RATE, epochs=33, layers='heads')

# print('Fine tune Resnet stage 4 and up')
# model.train(dataset_train, dataset_validation, learning_rate=config.LEARNING_RATE, epochs=120, layers='4+')

print('Fine tune all layers')
model.train(dataset_train, dataset_validation, learning_rate=config.LEARNING_RATE / 20, epochs=160, layers='all')

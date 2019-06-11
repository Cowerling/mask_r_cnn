from mrcnn import config
from mrcnn import models

from application.base.dataset import SpatialDataset


class TrainingMode(object):
    HEADS = 'heads'
    RESNET_4_UP = '4+'
    ALL = 'all'


def train(source, class_names, train_sources, validation_sources,
          weight_path, log_dir,
          epochs, training_mode, learning_rate_factor=1,
          images_per_gpu=5, gpu_count=1, image_dimension=(128, 128), confidence=0.9,
          steps_per_epoch=1000, validation_steps=50,
          name='TRAIN'):

    class TrainingConfig(config.Config):
        NAME = name

        IMAGES_PER_GPU = images_per_gpu
        GPU_COUNT = gpu_count

        IMAGE_MAX_DIM = image_dimension[0]
        IMAGE_MIN_DIM = image_dimension[1]

        DETECTION_MIN_CONFIDENCE = confidence

        NUM_CLASSES = len(class_names) + 1

        STEPS_PER_EPOCH = steps_per_epoch
        VALIDATION_STEPS = validation_steps

    dataset_train = SpatialDataset(source, class_names)
    dataset_validation = SpatialDataset(source, class_names)

    for train_source in train_sources:
        print('load train data: {}.{}'.format(train_source['rs_image_path'], train_source['database']))
        dataset_train.load_data(rs_image_path=train_source['rs_image_path'],
                                database=train_source['database'],
                                user=train_source['user'],
                                password=train_source['password'],
                                host=train_source['host'],
                                port=train_source['port'],
                                mask_table=train_source['mask_table'],
                                bound_table=train_source['bound_table'])

    for validation_source in validation_sources:
        print('load validation data: {}.{}'.format(validation_source['rs_image_path'], validation_source['database']))
        dataset_validation.load_data(rs_image_path=validation_source['rs_image_path'],
                                     database=validation_source['database'],
                                     user=validation_source['user'],
                                     password=validation_source['password'],
                                     host=validation_source['host'],
                                     port=validation_source['port'],
                                     mask_table=validation_source['mask_table'],
                                     bound_table=validation_source['bound_table'])

    dataset_train.prepare()
    dataset_validation.prepare()

    train_config = TrainingConfig()

    model = models.MaskRCNN(mode='training', config=train_config, model_dir=log_dir)
    model.load_weights(weight_path, by_name=True)

    print('start training')
    model.train(dataset_train, dataset_validation,
                learning_rate=train_config.LEARNING_RATE * learning_rate_factor,
                epochs=epochs,
                layers=training_mode)

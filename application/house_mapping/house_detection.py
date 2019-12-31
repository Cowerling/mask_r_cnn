import sys
import os


if __name__ == '__main__':
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))

    from application.base.remote_detection import remote_detect

    work_space = '/home/cowerling/DeepLearning/house/xiaozede'
    data_space = '/media/cowerling/share/data'

    remote_detect(host='10.6.72.250', port=9090,
                  class_names=['BG', 'house'],
                  weight_path=os.path.join(work_space, 'weight/mask_rcnn_house_gewei_0100.h5'),
                  log_dir=os.path.join(work_space, 'log'),
                  data_space=data_space,
                  bound_size=(1024, 1024), bound_buffer=(5, 5),
                  images_per_gpu=5, gpu_count=2)

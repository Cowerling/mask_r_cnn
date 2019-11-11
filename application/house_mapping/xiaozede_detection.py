import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))

from application.base.detection import detect

work_space = '/home/cowerling/DeepLearning/house_xiaozede'

detect(rs_image_path=os.path.join(work_space, 'image/xiaozede_sub.tif'),
       class_names=['BG', 'house'],
       weight_path=os.path.join(work_space, 'weight/mask_rcnn_house_gewei_0100.h5'),
       log_dir=os.path.join(work_space, 'log'),
       database='house',
       user='postgres',
       password='Cowerling',
       host='localhost',
       port=5432,
       mask_table='detect_house_xiaozede',
       block_table='detect_bound_xiaozeze',
       bound_size=(1024, 1024),
       bound_buffer=(5, 5))

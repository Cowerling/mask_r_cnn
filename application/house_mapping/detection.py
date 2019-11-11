import os

from application.base.detection import detect

work_space = '/home/cowerling/house_mapping'

detect(rs_image_path=os.path.join(work_space, 'image/honghu.tif'),
       class_names=['BG', 'house'],
       weight_path=os.path.join(work_space, 'weight/mask_rcnn_house-mapping_0160.h5'),
       log_dir=os.path.join(work_space, 'log'),
       database='house_mapping',
       user='postgres',
       password='Admin948',
       host='10.6.72.213',
       port=5432,
       mask_table='house',
       block_table='block',
       bound_size=(1024, 1024),
       bound_buffer=(5, 5))

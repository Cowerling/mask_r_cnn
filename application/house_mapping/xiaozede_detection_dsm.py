import os

from application.base.detection import detect

work_space = '/home/cowerling/house_mapping'

detect(rs_image_path=os.path.join(work_space, 'image/xiaozede_dsm_sub_normal_136_3.tif'),
       class_names=['BG', 'house'],
       weight_path=os.path.join(work_space, 'weight/mask_rcnn_gewei_dsm_0045.h5'),
       log_dir=os.path.join(work_space, 'log'),
       database='house_mapping',
       user='postgres',
       password='Admin948',
       host='10.6.72.213',
       port=5432,
       mask_table='xiaozede_dsm_house',
       block_table='xiaodeze_dsm_bound',
       bound_size=(1024, 1024),
       bound_buffer=(5, 5))

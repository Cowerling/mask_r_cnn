import os

from application.base.detection import detect

work_space = '/home/cowerling/house_mapping'

detect(rs_image_path=os.path.join(work_space, 'image/xiaozede_sub_c.tif'),
       class_names=['BG', 'house'],
       weight_path=os.path.join(work_space, 'weight/mask_rcnn_gewei_c_0100.h5'),
       log_dir=os.path.join(work_space, 'log'),
       database='house_mapping',
       user='postgres',
       password='Admin948',
       host='10.6.72.213',
       port=5432,
       mask_table='xiaozede_c_house',
       block_table='xiaodeze_c_bound',
       bound_size=(1024, 1024),
       bound_buffer=(5, 5),
       mean_pixel_values=[123.7, 116.8, 103.9, 32762.5])

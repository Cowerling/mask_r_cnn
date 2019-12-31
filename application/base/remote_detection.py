import json
from threading import Thread
import uuid
import psycopg2
import time
import os
from termcolor import colored

from application.base.tcp_server import TCPServer
from application.base.detection import detect


class DetectHandle(object):
    def __init__(self, class_names, weight_path, log_dir, data_space,
                 bound_size, bound_buffer, images_per_gpu, gpu_count):
        self.class_names = class_names
        self.weight_path = weight_path
        self.log_dir = log_dir
        self.data_space = data_space
        self.bound_size = bound_size
        self.bound_buffer = bound_buffer
        self.images_per_gpu = images_per_gpu
        self.gpu_count = gpu_count

    @staticmethod
    def change_job_status(connection_string, job_status_table, job_id, status, image=None, error=None):
        with psycopg2.connect(connection_string) as connection:
            with connection.cursor() as cursor:
                if status == 0:
                    cursor.execute('INSERT INTO {} (job_id, start_time, status, image) '
                                   'VALUES (\'{}\' , \'{}\', {}, \'{}\')'
                                   .format(job_status_table,
                                           job_id,
                                           time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                           status,
                                           image))
                elif status == 1:
                    cursor.execute('UPDATE {} SET end_time = \'{}\', status = {} WHERE job_id = \'{}\''
                                   .format(job_status_table,
                                           time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                           status,
                                           job_id))
                elif status == 2:
                    cursor.execute('UPDATE {} SET end_time = \'{}\', status = {}, error = \'{}\' WHERE job_id = \'{}\''
                                   .format(job_status_table,
                                           time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                           status,
                                           error,
                                           job_id))

    def process(self, job_id, parameter):
        connection_string = 'dbname={} user={} password={} host={} port={}'.format(parameter['task_status_db'],
                                                                                   parameter['db_user'],
                                                                                   parameter['db_password'],
                                                                                   parameter['db_host'],
                                                                                   parameter['db_port'])

        DetectHandle.change_job_status(connection_string, parameter['job_status'], job_id, 0, image=parameter['image'])

        try:
            detect(rs_image_path=os.path.join(self.data_space, parameter['image']),
                   class_names=self.class_names,
                   weight_path=self.weight_path,
                   log_dir=self.log_dir,
                   database=parameter['result_db'],
                   user=parameter['db_user'],
                   password=parameter['db_password'],
                   host=parameter['db_host'],
                   port=parameter['db_port'],
                   mask_table='mask_{}'.format(job_id),
                   block_table='block_{}'.format(job_id),
                   bound_size=self.bound_size,
                   bound_buffer=self.bound_buffer,
                   extent=parameter['extent'],
                   images_per_gpu=self.images_per_gpu, gpu_count=self.gpu_count,
                   tips=False)

            DetectHandle.change_job_status(connection_string, parameter['job_status'], job_id, 1)
        except Exception as e:
            DetectHandle.change_job_status(connection_string, parameter['job_status'], job_id, 2, error=e)
            print(colored('task failed: {}'.format(e), 'red'))

    def work(self, parameter):
        if 'image' not in parameter or \
                'result_db' not in parameter or \
                'task_status_db' not in parameter or \
                'db_user' not in parameter or \
                'db_password' not in parameter or \
                'db_host' not in parameter or \
                'db_port' not in parameter or \
                'job_status' not in parameter or \
                'extent' not in parameter:
            raise RuntimeError('parameter error')

        job_id = ''.join(str(uuid.uuid4()).split('-'))

        thread = Thread(target=self.process, args=(job_id, parameter,))
        thread.start()

        return json.dumps({'job_start': True, 'job_id': job_id})


def remote_detect(host, port, class_names, weight_path, log_dir, data_space,
                  bound_size, bound_buffer, images_per_gpu, gpu_count):
    tcp_server = TCPServer(host,
                           DetectHandle(class_names,
                                        weight_path, log_dir, data_space,
                                        bound_size, bound_buffer,
                                        images_per_gpu, gpu_count),
                           port)
    tcp_server.launch()

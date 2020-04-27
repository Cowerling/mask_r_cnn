FROM tensorflow/tensorflow:1.15.2-gpu-py3
RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN sed -i s@/security.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN apt-get clean
RUN add-apt-repository ppa:ubuntugis/ppa -y
RUN apt-get update
RUN apt-get install python-numpy gdal-bin libgdal-dev git -y
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple keras
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple rasterio
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple psycopg2
# gdalinfo --version
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple GDAL==2.4.2
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tqdm
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple imgaug
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scikit-image
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple Cython
RUN pip install git+https://github.com/waleedka/coco.git#subdirectory=PythonAPI

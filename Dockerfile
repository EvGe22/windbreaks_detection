FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-runtime

COPY requirements.txt requirements.txt

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN conda install gdal
RUN pip install -r requirements.txt
RUN apt-get install libsm6 libxext6 libxrender-dev

CMD "tail -f /dev/null"
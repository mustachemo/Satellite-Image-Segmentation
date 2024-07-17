# FROM tensorflow/tensorflow:latest-gpu # this 
# FROM tensorflow/tensorflow:latest
FROM nvcr.io/nvidia/tensorflow:24.06-tf2-py3

RUN rm -rf /workspace/*
WORKDIR /workspace

ADD requirements.txt .
RUN pip install --no-cache-dir --upgrade --pre pip
RUN pip install --no-cache-dir -r requirements.txt
ADD . .
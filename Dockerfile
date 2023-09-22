# Environment to use MicroCoral
#
# to build:
# docker build --force-rm -t microcoral -f .\Dockerfile .
#
# docker run --rm --name coral -it bash
#
FROM tensorflow/tensorflow:2.14.0rc1-jupyter

RUN pip3 install pip --upgrade
RUN pip3 install opencv-python opencv-contrib-python

WORKDIR /work
COPY code/ /work
COPY sh.test /work

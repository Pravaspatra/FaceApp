
FROM python:3.8-slim-buster
WORKDIR /Zenylog-Backend/
COPY ./requirements01.txt requirements01.txt
RUN set -xe \
    && apt-get update -y && apt-get install -y libgtk2.0-dev pkg-config
RUN apt-get install -y --fix-missing \
    build-essential \
    cmake \
    # gfortran \
    git \
    libgl1-mesa-glx \
    libglib2.0-dev \
    # wget \
    # curl \
    # graphicsmagick \
    # libgraphicsmagick1-dev \
    # libatlas-base-dev \
    # libavcodec-dev \
    # libavformat-dev \
    # libgtk2.0-dev \
    # libjpeg-dev \
    # liblapack-dev \
    # libswscale-dev \
    # pkg-config \
    python3-dev \
    python3-numpy \
    # software-properties-common \
    # zip \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*
RUN cd ~ && \
    mkdir -p dlib && \
    git clone -b 'v19.9' --single-branch https://github.com/davisking/dlib.git dlib/ && \
    cd  dlib/ && \
    python3 setup.py install --yes USE_AVX_INSTRUCTIONS
# Clone and build darknet
RUN git clone https://github.com/pjreddie/darknet.git \
    && cd darknet \
    && make
RUN python3 -m pip install --upgrade pip
RUN pip3 install face_recognition
#RUN pip3 uninstall opencv-python
RUN pip3 install opencv-python
RUN pip3 install gunicorn 
# RUN pip3 uninstall opencv-python-headless==4.5.2.52
# RUN pip3 uninstall opencv-contrib-python
RUN pip3 install -r  requirements01.txt
COPY . .
EXPOSE 8001
CMD ["gunicorn", "--bind", "0.0.0.0:8001", "zenylog.wsgi"]



# opencv-python==4.7.0.72
# sudo apt install libgtk2.0-dev pkg-config
# pip install opencv-python==4.0.0.
# git clone https://github.com/pjreddie/darknet.git
# cd darknet
# make
# pip install numpy

# FROM ubuntu:20.04
FROM nvcr.io/nvidia/tensorrt:21.07-py3
MAINTAINER Amiya Maity

# Clear lock files and old lists
RUN rm -rf /var/lib/apt/lists/* /var/cache/apt/* /var/lib/dpkg/lock* /var/lib/dpkg/lock-frontend /var/lib/dpkg/updates/*

# Update and install basic dependencies
RUN apt-get update && \
    apt-get install -y libopus0 fonts-dejavu-core libfontconfig1 && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata

# Reconfigure dpkg in case of any underlying issues
RUN dpkg --configure -a

# Install GStreamer packages in stages to identify potential conflicts
RUN apt-get update && \
    apt-get install -y \
        libgstreamer1.0-0 \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good && \
    apt-get install -f -y && \
    dpkg --configure -a && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install remaining GStreamer packages in a separate step
RUN apt-get update && \
    apt-get install -y \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-libav \
        gstreamer1.0-doc \
        gstreamer1.0-tools \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-dev && \
    apt-get install -f -y && \
    dpkg --configure -a && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*





RUN apt-get install -y git

# just for testing
RUN apt-get -y install nano

# setup python
RUN apt-get update && \
    apt-get install -y \
	python3-dev python3-pip \
	gcc libxml2-dev libxslt-dev



RUN pip3 install numpy future lxml pymavlink

# get opencv and build it
RUN git clone https://github.com/opencv/opencv.git

RUN apt-get install -y build-essential libssl-dev

RUN apt-get -y install cmake libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavresample-dev

RUN \
	cd opencv && \
	git checkout 4.5.4 && \
	git submodule update --recursive --init && \
	mkdir build && \
	cd build && \
	cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D INSTALL_C_EXAMPLES=OFF \
	-D PYTHON_EXECUTABLE=$(which python3) \
	-D BUILD_opencv_python2=OFF \
	# -D CMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
	-D CMAKE_INSTALL_PREFIX=/opt/opencv \
	-D PYTHON3_EXECUTABLE=$(which python3) \
	-D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
	-D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
	-D WITH_FFMPEG=ON \
    -D WITH_GSTREAMER=ON \
	-D BUILD_EXAMPLES=ON .. && \
	make -j$(nproc) && \
	make install && \
	ldconfig



RUN apt-get update && \
    apt-get install -y \
    libspdlog-dev libjson-c-dev libboost-all-dev


RUN wget https://sourceforge.net/projects/crow-framework.mirror/files/v1.0%2B5/crow-v1.0%2B5.deb
RUN dpkg -i crow-v1.0+5.deb \
    && pip3 install Pillow numpy flask requests

RUN mkdir -p /opt/NlohmannJSON/include \
&& cd /opt/NlohmannJSON/include \
&& wget https://raw.githubusercontent.com/nlohmann/json/v3.8.0/single_include/nlohmann/json.hpp


WORKDIR /app
COPY . /app/

RUN cd /app/ \
&& mkdir -p build \
&& cd build \
&& cmake .. \
&& make \
&& ldconfig

RUN cp /app/build/trtDetection /app/ \
    && chmod 777 /app/scripts/run.sh \
    && cp /app/scripts/run.sh /app/

ENTRYPOINT ["./run.sh"]

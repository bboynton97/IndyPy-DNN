# Pull base image.
FROM ubuntu

# install the bare minimum packages needed to run
RUN \
  apt-get update && \
  apt-get -y upgrade && \
  apt-get install -y python2.7 python-pip vim && \
  rm -rf /var/lib/apt/lists/*

# Make python2.7 and dependencies happy
RUN \
  pip install --upgrade pip && \
  pip install tensorflow && \
  pip install keras && \
  pip install pandas && \
  pip install numpy && \
  pip install scipy && \
  pip install sklearn

# Add files.
ADD BreastCancerDNN.py /root/BreastCancerDNN.py
ADD breast-cancer-wisconsin.csv /root/breast-cancer-wisconsin.csv

# Set environment variables.
ENV HOME /root

# Define working directory.
WORKDIR /root

# Define default command.
CMD [ "python","BreastCancerDNN.py"]


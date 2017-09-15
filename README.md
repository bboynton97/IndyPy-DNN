# Breast Cancer Identifier

## Environment Setup
macOS or Linux is recommended. Tensorflow now has support for Windows, but they still don’t play well together.

#### Install Python2.7 or 3

I will be using Python2.7 in the tutorial.<br />
Use `python --version` to check your install

If you don’t have Python, you can download it here: https://www.python.org/downloads/

#### Install PIP
Mac: `sudo easy_install pip`<br />
Linux: `sudo apt-get install python-pip python-dev`<br />
Windows: It looked complicated for Windows, so I’m just gunna hope you have it already

#### Install TensorFlow
*Do not use the GPU-enabled version unless you know what you’re doing*<br />
`pip install --user tensorflow`

#### Install Keras
`pip install --user keras`

#### Install Keras Dependencies
`pip install --user numpy`

#### Install Metrics Library
`pip install --user sklearn`

#### Install preprocessing tools
`pip install --user pandas`

### Copy-paste this line for all:
`pip install --user tensorflow && pip install --user keras && pip install --user numpy && pip install --user scipy && pip install --user pandas && pip install --user sklearn`

## Test the libraries
python

import tensorflow

import keras

import sklearn

import pandas

## Get our dataset
Download the breast cancer dataset from:<br />
http://www.vemity.com/wp-content/uploads/2017/08/breast-cancer-wisconsin.csv<br />
Right click the link and choose “Download File”

Alternatively (if the link is printed): Copy-paste the contents of the page into a text editor and save it somewhere as breast-cancer-wisconsin.csv

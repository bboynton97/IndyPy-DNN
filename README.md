#Breast Cancer Identifier

##Environment Setup
macOS or Linux is recommended. Tensorflow now has support for Windows, but they still don’t play well together.

Install Python2.7 or 3
I will be using Python2.7 in the tutorial
`python --version` to check your install

If you don’t have Python, you can download it here: https://www.python.org/downloads/

Install PIP
Mac: `sudo easy_install pip`
Linux: `sudo apt-get install python-pip python-dev`
Windows: It looked complicated for Windows, so I’m just gunna hope you have it already

Install TensorFlow
*Do not use the GPU-enabled version unless you know what you’re doing*
`sudo pip install tensorflow`

Install Keras
`sudo pip install keras`

Install Keras Dependencies
`sudo pip install numpy`
`sudo pip install scipy`

Get our dataset
Download the breast cancer dataset from:
http://www.vemity.com/wp-content/uploads/2017/08/breast-cancer-wisconsin.csv
Right click the link and choose “Download File”

Alternatively (if the link is printed): Copy-paste the contents of the page into a text editor and save it somewhere as breast-cancer-wisconsin.csv

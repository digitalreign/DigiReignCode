# Convolutional Neural Networks with TensorFlow in Python
Advanced neural networks: Master Computer Vision with Convolutional Neural Networks (CNN) and Deep Learning

## Prerequisites
Anaconda

### Anaconda For Ubuntu
apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6

download the anaconda package from https://www.anaconda.com/download/#linux

bash ~/Downloads/Anaconda3-2020.02-Linux-x86_64.sh \
WITH THE NAME OF THE PACKAGE YOU DOWNLOADED \
You may have to install with the -u option to update the existing package.


### Anaconda for Windows
Install python 3 from the windows store. It is easiest to do this by running 'python' from the powershell as it will open the microsoft store for you.

Download the Anaconda Package \
https://www.anaconda.com/products/individual#Windows

In Windows, you will have to set the path to the location where you installed Anaconda3 to.

For me, I installed anaconda3 into C:\Anaconda3. Therefore you need to add C:\Anaconda3, C:~\Python\Python38\Scripts,  as well as C:\Anaconda3\Scripts\ to your path variable, e.g. set PATH=%PATH%;C:\Anaconda3;C:\Anaconda3\Scripts\.

You can do this via powershell (see above, https://msdn.microsoft.com/en-us/library/windows/desktop/bb776899(v=vs.85).aspx ), or hit the windows key → enter environment → choose from settings → edit environment variables for your account → select Path variable → Edit → New.

To test it, open a new dos shell, and you should be able to use conda commands now. E.g., try conda --version.


### Visual Studio Code Setup
https://code.visualstudio.com/docs/languages/python
https://medium.com/@udiyosovzon/how-to-activate-conda-environment-in-vs-code-ce599497f20d \
from an Administrator Powershell prompt change the Powershell Execution Policy to remote signed i.e. Set-ExecutionPolicy RemoteSigned \
open an Anaconda Prompt and run conda init powershell which will add Conda related startup to a Powershell profile.ps1 somewhere in your user's profile.


### Install Tensorflow
pip install tensorflow --user \
pip install tensorflow-datasets --user \
pip install tf-nightly-gpu --user \


#### If you plan to use CUDA then go through the CUDA instructions here:
https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/

#### Download and install Visual Studio 2017 and 2019:
https://my.visualstudio.com/Download 

#### Download the CUDA Toolkit 11.2.
Depending on your video card you may need some or all of these. I would suggest running a program that calls for tensorflow to pull from the GPU and see what errors you get, you are looking for .dlls that end in 8, 10, 110, and 11 to say if you need 8, 10.1 or 11.2 to run. \
I would also highly recommend installing them lowest number to highest number.

8.0 \
https://developer.nvidia.com/cuda-80-ga2-download-archive 

Then 10.1: \
https://developer.nvidia.com/cuda-10.1-download-archive-base 

Then 11.2: \
https://developer.nvidia.com/cuda-downloads

cuDNN for your CUDA version. \
I placed the cuda directory in the base C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\ \
https://developer.nvidia.com/rdp/cudnn-archive


Add the following to your ENV in Windows depending on what versions you need: \
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin \
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\extras\CUPTI\lib64 \
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include \
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin \
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\extras\CUPTI\lib64 \
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include \
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin \
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\extras\CUPTI\lib64 \
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include 

You have to include this one: \
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\cuda\bin \
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\cuda\include \
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\cuda\lib\x64

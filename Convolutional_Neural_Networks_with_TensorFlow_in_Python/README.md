# Convolutional Neural Networks with TensorFlow in Python
Advanced neural networks: Master Computer Vision with Convolutional Neural Networks (CNN) and Deep Learning

## Prerequisites
Anaconda

### Anaconda For Ubuntu
apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6

download the anaconda package from https://www.anaconda.com/download/#linux

bash ~/Downloads/Anaconda3-2020.02-Linux-x86_64.sh \
WITH THE NAME OF THE PACKAGE YOU DOWNLOADED\
You may have to install with the -u option to update the existing package.


### Anaconda for Windows
Install python 3 from the windows store. It is easiest to do this by running 'python' from the powershell as it will open the microsoft store for you.

Download the Anaconda Package 
https://www.anaconda.com/products/individual#Windows

In Windows, you will have to set the path to the location where you installed Anaconda3 to.

For me, I installed anaconda3 into C:\Anaconda3. Therefore you need to add C:\Anaconda3, C:~\Python\Python38\Scripts,  as well as C:\Anaconda3\Scripts\ to your path variable, e.g. set PATH=%PATH%;C:\Anaconda3;C:\Anaconda3\Scripts\.

You can do this via powershell (see above, https://msdn.microsoft.com/en-us/library/windows/desktop/bb776899(v=vs.85).aspx ), or hit the windows key → enter environment → choose from settings → edit environment variables for your account → select Path variable → Edit → New.

To test it, open a new dos shell, and you should be able to use conda commands now. E.g., try conda --version.


### Visual Studio Code Setup
https://code.visualstudio.com/docs/languages/python
https://medium.com/@udiyosovzon/how-to-activate-conda-environment-in-vs-code-ce599497f20d \
from an Administrator Powershell prompt change the Powershell Execution Policy to remote signed i.e. Set-ExecutionPolicy RemoteSigned\
open an Anaconda Prompt and run conda init powershell which will add Conda related startup to a Powershell profile.ps1 somewhere in your user's profile.


### Install Tensorflow
pip install tensorflow --user\
pip install tensorflow-datasets --user

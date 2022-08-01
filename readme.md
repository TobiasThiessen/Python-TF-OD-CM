# Custom Model for Object Detection powered by Tensorflow
## General Knowledge Prerequisites:
- Object Detection
- Machine Learning
- Python and programming
- Virtual Environments

## Program Prerequisites:
Known to work:
- Python 3.10.4
- CUDA 11.2 
- CudNN 8.1.1
- PIP 22.1.2
- GIT 2.26.1.windows.1
- more

Known to not work:
- CUDA 11.7
- cudNN 8.4.1

Downloads:
- Python download:  https://www.python.org/downloads/
- CUDA download:    https://developer.nvidia.com/cuda-downloads (may require nvidia dev account (free))
- cuDNN download:   https://developer.nvidia.com/rdp/cudnn-download (requires nvidia dev account (free))
- GIT download:     https://git-scm.com/download/win

# Installation / Setup:
#### Step 1: Install prerequisites
Follow the guides from the various downloaders, or use my quick guides:

###### Python: 
- Run the exe.
- Check if python is installed by typing in cmd or shell:
> Python

###### CUDA:
- Run the exe.
- Check if CUDA 11.2 is installed by typing in cmd or shell:
> nvcc --version

###### cuDDN:
- Copy files into CUDA directory.
- Check if cuDNN 8.1.1 is installed by reading the following header file:
> NVIDIA GPU Computing Toolkit\CUDA\v11.2\include\cudnn_version.h

###### PIP:
- Should have been installed through the python installer.
- Check if pip is installed by typing in cmd or shell:
> pip --version

###### GIT:
- Run the exe.
- Check if GIT is installed by typing in cmd or shell:
> git --version

#### Step 2: Run "environment_setup.py"
Use whatever method to run the script, i.e. from IDE or CMD.
This should automatically make a virtual environment named "VENV" and install all the necesarry packages from "library_requirements.txt".

# How to use: A guide
#### The workflow: Quickguide
1. Obtain images with objects to detect.
2. Use the labeling tool to label objects in images - save label file with image, use same name for ease.
3. Update labels in "pythonfilename.py" to match labels from previous step.
4. From main.py uncomment a model

# Ideal Improvements:
#### Write_Records
Should only be called when image numbers or labels change.

#### Stop using OS to call CMD
Very inefficient and hard to keep track.

# Extra Information:
#### How to label:
> python labeling.py

The script should ideally be ran in the VENV of the project, as it will already have the installed modules necessary.

#### Model Zoo:
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

Known to work:
- SSD MobileNet v2 320x320
- SSD MobileNet V2 FPNLite 640x640
- EfficientDet D0 512x512

Known to not work:
- CenterNet Resnet50 V2 512x512
- CenterNet Resnet50 V1 FPN 512x512

#### Hardware known to function
Processors:
- AMD Ryzen 5 3600X 

GPUs:
- NVidia Geforce GTX 1060 6GB (Driver Version: 516.01)

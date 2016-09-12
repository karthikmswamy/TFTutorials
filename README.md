# TFTutorials

Prerequisites
-------------------------------
Basic understanding of Python
Familiarity with basic machine learning concepts (good to have but not required)

Materials and Packages
-------------------------------
A laptop (Ubuntu, Mac OSX, Windows (10 preferred))
Installation of following packages:
	- Python 2.7
	- TensorFlow
	- TFLearn

This tutorial requires installation of TensorFlow and TFLearn to understand how both systems work.

Installation of both TensorFlow and TFLearn is fairly straightforward if you're on Ubuntu or Mac OSX. 
You can skip to Install Instructions if you're on Ubuntu or Mac OSX.

Installing on Windows 10
-------------------------------
Windows 10 Anniversary Edition (released in August 2016) comes bundled with Bash on Ubuntu on Windows. If you have a Windows 10 machine, get the Anniversary Update following instructions here: https://blogs.windows.com/windowsexperience/2016/08/02/how-to-get-the-windows-10-anniversary-update/
Follow the install instructions to get Bash on Ubuntu on Windows: https://msdn.microsoft.com/commandline/wsl/install_guide
Once you have an Ubuntu bash running on your Windows machine do a sudo apt-get update and follow Install Instructions below.

Installing on Windows 8, 7 or earlier
------------------------------------------
TensorFlow is not natively supported on Windows yet. You might need to install Ubuntu 14.04 on a VirtualBox and follow the instructions for Ubuntu below.
VirtualBox for Windows: http://download.virtualbox.org/virtualbox/5.1.4/VirtualBox-5.1.4-110228-Win.exe
Ubuntu 14.04 ISO: http://releases.ubuntu.com/14.04/ubuntu-14.04.4-desktop-amd64.iso

Install Instructions
-------------------------------
Follow the link below for installing TensorFlow based on the type of install:
If you are the only one using TensorFlow, the easiest method to install is using pip. 
If you would like to test TensorFlow before a full-fledged install, use the docker-based or virtualenv-based installation.

Pip-based install: https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html#pip-installation
Virtualenv-based install: https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html#virtualenv-installation
Docker-based install: https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html#docker-installation

To install TFLearn you need to install TensorFlow. Once you've installed TensorFlow, follow the instructions in the link below to install TFLearn:
http://tflearn.org/installation/#tflearn-installation

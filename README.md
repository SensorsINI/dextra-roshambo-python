The Dextra rock-scissor-paper robot perception pipeline redone in python using pyaer and tensorflow.
See also
 * The Dextra page https://sensors.ini.ch/research/projects/dextra
 * The Dextra tendon driven hand paper and design https://sensorsini.github.io/dextra-robot-hand/
 * The Arduino firmware [https://github.com/SensorsINI/DextraRoshamboHand](https://github.com/SensorsINI/Dextra-robot-hand-firmware)
 * The ROSHAMBO17 dataset https://docs.google.com/document/d/1rOltN_BaOTAMbP1chzFZxPjN24eTdbzuCrCM4S2o6qA/edit?tab=t.0 

The pretrained network is a 16-bit quantized weight and state CNN trained on the dataset 

## Requirements

 - OS: Fully tested on Ubuntu 18.04 and 22.04
  * Python 3.9 **Important, necessary for Tensorflow 2.5.0 required to run the roshambo CNN**
  * Tensorflow 2.5.0
  * CUDA 10.2+ or whatever comes with tensorflow install
    - Keras: 2.5.0
    - pyaer https://github.com/duguyue100/pyaer
 
 * sensors DAVIS camera
 * Robot hand with Arduino control via USB serial port bytes if you want to see the hand move; see project https://github.com/SensorsINI/DextraRoshamboHand

## Setup

Project includes pycharm _.idea/_ folder and vscode _.vscode/_ folder.

### System level prerequisites
1. *Install libcaer.* Can be installed with  sudo apt-get install libcaer-dev, otherewise see https://gitlab.com/inivation/dv/libcaer.

### Make a conda environment
Create the environment, activate it, then in it install the libraries. We recommend you use conda because it will download the necessary python version 3.9. (That
is the last python version to have tensorflow 2.5.0 which this project uses.) 
``` bash
conda create -n roshambo python=3.9
conda activate roshambo
```
### Other requirements
See requirements.txt for libraries needed. 
Install them to the new conda env from the conda prompt with

``` bash
conda activate roshambo # probably already activate
pip install -r requirements.txt
```

If you have trouble with pyaer, see https://github.com/duguyue100/pyaer. It should work for linux and mac OS intel silicon machines. Windows probably will not work natively, but you can run the code in a WSL2 Ubuntu virtual machine using https://github.com/dorssel/usbipd-win to map the USB port to WSL2 Ubuntu.

### USB/pyaer on Windows

Recommand to use WSL2 and the vscode plugin https://marketplace.visualstudio.com/items?itemName=thecreativedodo.usbip-connect to bind the USB port to WSL2 Ubuntu VM

## Museum kiosk requirements
For unattended operation, it is necessary that rtcwake can suspend the computer so that the user permissions allow it over reboots.
* Copy [9-userdev-input.rules](99-userdev-input.rules) into /etc/udev/rules.d
* Copy [power-state.conf](power-state.conf) into /etc/tmpfiles.d
* Copy [dextra.desktop](dextra.desktop) and [symbols/dextra-icon.png](symbols/dextra-icon.png) to ~/.local/share/applications (for desktop launcher) and to ~/.config/autostart, so login launches demo
* Set up user for autologin
These files may need editing for your username.

# Running Dextra

Run _roshambo_; it uses multiprocessing to launch 2 subprocessees, _producer_ and _consumer_. (You can run these separately for testing.)

Run two processes, producer and consumer.

 1. connect hardware: DAVIS to USB and Arduino to USB.
 2. Find out which serial port device the hand controller Arduino appears on. You can use dmesg on linux. You can put the serial port into _globals_and_utils.py_ to avoid adding as argument.
 2. In terminal run producer
```bash
python -m roshambo
```
 


The Dextra rock-scissor-paper robot redone in python using pyaer and tensorflow.

The pretrained network is a 16-bit quantized weight and state CNN.

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
pip install -f requirements.txt
```

If you have trouble with pyaer, see https://github.com/duguyue100/pyaer

# Running Dextra

Run roshambo; it uses multiprocessing to launch 2 subprocessees, _producer_ and _consumer_. (You can run these separately for testing.)

Run two processes, producer and consumer.

 1. connect hardware: DAVIS to USB and Arduino to USB.
 2. Find out which serial port device the hand controller Arduino appears on. You can use dmesg on linux. You can put the serial port into _globals_and_utils.py_ to avoid adding as argument.
 2. In terminal run producer
```bash
python -m roshambo
```
 


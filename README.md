The Dextra robot redone in python using pyaer and tensorflow.

The pretrained network is a 16-bit quantized weight and state CNN.

## Requirements

 - OS: Fully tested on Ubuntu 18.04
  * Python 3.9
 * Tensorflow 2.5.0
 * CUDA 10.2+
 - Keras: 2.5.0
 - pyaer https://github.com/duguyue100/pyaer
 
 * sensors DVS camera
 * Robot hand with Arduino control via USB serial port bytes

## Setup

Project includes pycharm .idea files.

**Make a conda environment**, activate it, then in it install the libraries.
``` bash
conda create -n roshambo python=3.9
conda activate roshambo
```
### pyaer
pyaer needs https://gitlab.com/inivation/dv/libcaer. 

Clone it, then follow instructions in its README to install libcaer. 

### Other requirements
See requirements.txt for libraries needed. 
Install them to the new conda env from the conda prompt with

``` bash
conda activate roshambo # probably already activate
pip install -f requirements.txt
```/


# Running Dextra

Run two processes, producer and consumer.

 1. connect hardware: DVS to USB and Arduino to USB.
 1. Find out which serial port device the Arduino appears on. You can use dmesg on linux. You can put the serial port into _globals_and_utils.py_ to avoid adding as argument.
 1. In first terminal run producer
```shell script
python -m producer
```
 2. In a second terminal, run consumer
```shell script
python -m consumer  arduinoPort
example: python -m consumer.py 
```



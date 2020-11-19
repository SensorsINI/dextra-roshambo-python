The Dextra robot redone in python using pyaer and tensorflow.

The pretrained network is a 16-bit quantized weight and state CNN.

## Requirements

 - OS: Fully tested on Ubuntu 18.04
  * Python 3.8
 * Tensorflow 2.3.1
 * CUDA 10.2+
 - Keras: 2.3.1
 - pyaer https://github.com/duguyue100/pyaer
 
 * sensors DVS camera
 * Robot hand with Arduino control via USB serial port bytes

## Setup
See requirements.txt for libraries needed. 

Project includes pycharm .idea files.

**Make a conda environment**, activate it, then in it install the libraries.

Install them to a new conda env with pip install -f requirements.txt from the conda prompt.

```
pip install opencv-python tensorflow keras pyserial pyaer engineering_notation matplotlib 
```
### pyaer
pyaer needs https://github.com/inivation/libcaer. Clone it, then follow instructions in its README to install libcaer. 


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



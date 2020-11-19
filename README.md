The Dextra robot redone in python using pyaer and tensorflow.

The pretrained network is a 16-bit quantized weight and state CNN.

## Requirements

 * Python 3.8
 * Tensorflow 2.3.1
 * CUDA 10.2+
 * sensors DVS camera
 * Robot hand with Arduino control via USB serial port bytes

## Setup
See requirements.txt for libraries needed. 

Project includes pycharm .idea files.

Install them to a new conda env with pip install -f requirements.txt from the conda prompt.

# Running Dextra

Run two processes, producer and consumer.




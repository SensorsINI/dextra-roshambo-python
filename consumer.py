"""
consumer of DVS frames for classification of DVS frames
Authors: Tobi Delbruck, Nov 2020
"""
import argparse
import copy
import glob
import pickle
import cv2
import sys
import keras.saving
import keras.saving
import tensorflow as tf
# from keras.models import load_model

import serial
import socket
from select import select
import multiprocessing.connection as mpc
from multiprocessing import  Pipe,Queue

from tensorflow.python.keras import Input

from RoshamboNet import RoshamboNet
from globals_and_utils import *
from engineering_notation import EngNumber  as eng # only from pip
import collections
from pathlib import Path
import random

from tensorflow.python.keras.models import load_model, Model
from Quantizer import apply_quantization
log=my_logger(__name__)
from numpy_loader import load_from_numpy

# Only used in mac osx
try:
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
except Exception as e:
    print(e)

class majorityVote:
    #filter cmd with majority vote
    def __init__(self, window_length, num_classes): # window_length is size of window in votes, num_classes is the number of possible values, 0 to num_classes-1
        self.window_length = window_length
        self.num_classes = num_classes
        self.ptr = 0 # pointer to circular buffer
        self.cirbuf = np.full(self.window_length, -1, dtype=np.int8) # cirular buffer of most recent predictions
        self.cmdcnts = np.zeros(num_classes, dtype=np.int8) # hold the number of votes for each prediction
        self.num_predictions=0

    def new_prediction_and_vote(self, symbol): # cmd is the new value, in range 0 to num_classes-1
        """ Takes new prediction of symbol, returns possible new vote
        :param symbol: the new classification of hand symbol
        :returns: the majority vote or None if there is no majority
        """
        if 0 <= symbol < self.num_classes:
            self.num_predictions+=1
            idx = self.ptr # pointer to current idx in circular buffer
            if self.num_predictions>self.window_length: 
                self.cmdcnts[self.cirbuf[idx]] -= 1 # decrement count for previous prediction but only if we already filled the buffer, otherwise we end up with negative background
            self.cirbuf[idx] = symbol # store latest prediction
            self.cmdcnts[symbol] += 1  # vote for this prediction

            self.ptr = (self.ptr + 1) % self.window_length # increment and wrap pointer

        return self.vote()

    def vote(self): 
        """ produces the majority vote
        :returns: the majority if there is one, otherwise None
        """
        majority_count = self.window_length // 2 + 1 # e.g. 3 for window_length=5
        imax = np.argmax(self.cmdcnts)
        if self.cmdcnts[imax] >= majority_count:
            return imax
        return None


cmdVoter = majorityVote(5, 4)
useMajority= True

def classify_img(img: np.array, interpreter, input_details, output_details):
    """ Classify uint8 img

    :param img: input image as unit8 np.array range 0-255
    :param interpreter: the TFLITE interpreter
    :param input_details: the input details of interpreter
    :param output_details: the output details of interpreter

    :returns: symbol ('background' 'rock','scissors', 'paper'), class number (0-3), softmax output vector [4]
    """
    interpreter.set_tensor(input_details[0]['index'], (1/256.)*np.array(np.reshape(img, [1, IMSIZE, IMSIZE, 1]), dtype=np.float32))
    interpreter.invoke()
    pred_vector = interpreter.get_tensor(output_details[0]['index'])[0]
    pred_idx=np.argmax(np.array(pred_vector))
    pred_class_name=list(CLASS_DICT.keys())[list(CLASS_DICT.values()).index(pred_idx)]
    return pred_class_name, pred_idx, pred_vector


def load_latest_model_convert_to_tflite():

    input_tensor=Input(shape=(IMSIZE, IMSIZE, 1))
    x = RoshamboNet(
        input_tensor,
        classes=4,
        include_top=True,
        pooling="avg",
        num_3x3_blocks=3,
        )
    model = Model(inputs=input_tensor, outputs=x, name='roshambo')
    # model=apply_quantization(model, pruning_policy=None, weight_precision=16, activation_precision=16,
    #                    activation_margin=None)
    # model.load_weights(os.path.join(MODEL_DIR, MODEL_BASE_NAME))
    load_from_numpy(model,'model/numpy_weights')
    print(f'model.input_shape: {model.input_shape}')
    model.save('roshambo-model',save_format='tf')
    log.info('converting model to tensorflow lite model')
    converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)  # path to the SavedModel directory
    tflite_model = converter.convert()
    tflite_model_path = os.path.join(MODEL_DIR, TFLITE_FILE_NAME)

    log.info(f'saving tflite model as {tflite_model_path}')
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    return model, tflite_model


def load_tflite_model(folder=None):
    """ loads the most recent trained TFLITE model

    :param folder: folder where TFLITE_FILE_NAME is to be found, or None to find latest one

    :returns: interpreter,input_details,output_details

    :raises: FileNotFoundError if TFLITE_FILE_NAME is not found in folder
    """
    tflite_model_path=None
    if folder is None:
        existing_models = glob.glob(MODEL_DIR + '/' + DEXTRA_NET_BASE_NAME + '_*/')
        if len(existing_models) > 0:
            latest_model_folder = max(existing_models, key=os.path.getmtime)
            tflite_model_path = os.path.join(latest_model_folder, TFLITE_FILE_NAME)
            if not os.path.isfile(tflite_model_path):
                raise FileNotFoundError(f'no TFLITE model found at {tflite_model_path}')
        else:
            raise FileNotFoundError(f'no models found in {MODEL_DIR}')

    else:
        tflite_model_path=os.path.join(folder, TFLITE_FILE_NAME)
        log.info('loading tflite CNN model {}'.format(tflite_model_path))
    # model = load_model(MODEL)
    # tflite interpreter, converted from TF2 model according to https://www.tensorflow.org/lite/convert

    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details

def consumer(queue:Queue):
    """
    consume frames to predict polarization
    :param queue: if started with a queue, uses that for input of voxel volume
    """

    def none_or_str(value):
        if value == 'None':
            return None
        return value

    parser = argparse.ArgumentParser(
        description='consumer: Consumes DVS frames for trixy to process', allow_abbrev=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--serial_port", type=none_or_str, default=SERIAL_PORT,
        help="serial port, e.g. /dev/ttyUSB0 or None to not user port")

    args = parser.parse_args()

    log.info('opening UDP port {} to receive frames from producer'.format(PORT))
    server_socket: socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    address = ("", PORT)
    server_socket.bind(address)
    load_latest_model_convert_to_tflite()
    interpreter, input_details, output_details=load_tflite_model(MODEL_DIR)

    arduino_serial_port=None
    serial_port = args.serial_port
    if not serial_port is None:
        log.info('opening serial port {} to send commands to finger'.format(serial_port))
        try:
            arduino_serial_port = serial.Serial(serial_port, 115200, timeout=5)
        except Exception as e:
            log.error(f'could not open serial port to control hand - ignoring ({e})')
    log.info(f'Using UDP buffer size {UDP_BUFFER_SIZE} to recieve the {IMSIZE}x{IMSIZE} images')

    STATE_IDLE = 0
    STATE_FINGER_OUT = 1
    state = STATE_IDLE

    log.info('GPU is {}'.format('available' if len(tf.config.list_physical_devices('GPU')) > 0 else 'not available (check tensorflow/cuda setup)'))


    def show_frame(frame, name, resized_dict):
        """ Show the frame in named cv2 window and handle resizing

        :param frame: 2d array of float
        :param name: string name for window
        """
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, frame)
        if not (name in resized_dict):
            cv2.resizeWindow(name, 300, 300)
            resized_dict[name] = True
            # wait minimally since interp takes time anyhow
            cv2.waitKey(1)


    last_frame_number=0
    cv2_resized=False
    while True:
        timestr = time.strftime("%Y%m%d-%H%M")
        # with Timer('overall consumer loop', numpy_file=f'{DATA_FOLDER}/consumer-frame-rate-{timestr}.npy', show_hist=True):
        with Timer('overall consumer loop', numpy_file=None, show_hist=False):
            with Timer('recieve UDP'):
                receive_data = server_socket.recv(UDP_BUFFER_SIZE)

            with Timer('unpickle and normalize/reshape'):
                (frame_number,timestamp, img) = pickle.loads(receive_data)
                dropped_frames=frame_number-last_frame_number-1
                if dropped_frames>0:
                    log.warning(f'Dropped {dropped_frames} frames from producer')
                last_frame_number=frame_number
                # img = (1./255)*np.reshape(img, [IMSIZE, IMSIZE,1])
            with Timer('run CNN', numpy_file=None, show_hist=True):
                # pred = model.predict(img[None, :])
                pred_class_name, pred_idx, pred_vector=classify_img(img, interpreter, input_details, output_details)

            if pred_idx<=3: # symbol recognized (or background==3)
                #
                if useMajority:
                    f_cmd = cmdVoter.new_prediction_and_vote(pred_idx)
                    if not (f_cmd is None):
                        pred_idx = f_cmd
                        # find majority class name since it might be different than most recent prediction
                        pred_class_name=list(CLASS_DICT.keys())[list(CLASS_DICT.values()).index(pred_idx)]
                        if not arduino_serial_port is None:
                            if pred_idx==0:
                                arduino_serial_port.write(b'2')
                            elif pred_idx==1:
                                arduino_serial_port.write(b'3')
                            elif pred_idx==2:
                                arduino_serial_port.write(b'1')
                        
                        # arduino_serial_port.write(pred_idx)
                elif not arduino_serial_port is None:
                    arduino_serial_port.write(pred_idx)


            cv2.putText(img, pred_class_name, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

            cv2.namedWindow('RoshamboCNN', cv2.WINDOW_NORMAL)
            cv2.imshow('RoshamboCNN', 1 - img.astype('float') / 255)
            if not cv2_resized:
                cv2.resizeWindow('RoshamboCNN', 600, 600)
                cv2_resized = True
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q') or k==ord('x'):
                break
            elif k == ord('p'):
                print_timing_info()

            # save time since frame sent from producer
            dt=time.time()-timestamp
            with Timer('producer->consumer inference delay',delay=dt, show_hist=True):
                pass

if __name__ == '__main__':
    consumer(queue=None)

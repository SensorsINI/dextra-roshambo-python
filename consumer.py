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
from datetime import datetime # for hour of day for running demo 

from tensorflow.python.keras.models import load_model, Model
# from Quantizer import apply_quantization
log=my_logger(__name__)
from numpy_loader import load_from_numpy



# Only used in mac osx
try:
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
except Exception as e:
    print(e)

class majority_vote:
    #filter cmd with majority vote
    def __init__(self, window_length, num_classes): # window_length is size of window in votes, num_classes is the number of possible values, 0 to num_classes-1
        """ Does median filter majority vote over past predictions of human hand symbol
        :param window length: the size of window in votes
        :param num_classes: the number of possible values, 0 to num_classes-1
        :return: the majority if there is one, else None
        """
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
    time_last_sent_cmd=time.time()
    time_last_showed_demo_movement=time.time()
    arduino_serial_port=None

    def none_or_str(value):
        if value == 'None':
            return None
        return value

    def maybe_show_demo_sequence():
        time_now=datetime.now().time()
        if time_now>MUSEUM_OPENING_TIME and time_now<MUSEUM_CLOSING_TIME:
            now=time.time()

            nonlocal time_last_showed_demo_movement
            nonlocal time_last_sent_cmd
            time_interval=now-time_last_showed_demo_movement
            if time_interval>MUSEUM_HAND_MOVEMENT_INTERVAL_M*60 \
                    and now-time_last_sent_cmd>MUSEUM_HAND_MOVEMENT_INTERVAL_M*60:
                
                time_last_showed_demo_movement=now
                
                log.debug(f'showing demo movement because {MUSEUM_HAND_MOVEMENT_INTERVAL_M} minutes since last demo movement')
                show_demo_sequence()


    def show_demo_sequence():
        log.debug('showing demo sequence')
        cmds=[b'3',b'2',b'1'] # 3=rock, 1=paper, 2=scissors
        interval_seconds=.6

        try:
            for c in cmds:
                arduino_serial_port.write(c)
                log.debug(f'sent {c}, sleeping {interval_seconds}s')
                time.sleep(interval_seconds)

        except serial.serialutil.SerialException as e:
            log.error(f'Error writing to serial port {SERIAL_PORT} with cmd {cmd} for detected symbol {pred_class_name}: {e}')
                

    parser = argparse.ArgumentParser(
        description='consumer: Consumes DVS frames for trixy to process', allow_abbrev=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--serial_port", type=none_or_str, default=SERIAL_PORT,
        help="serial port, e.g. /dev/ttyUSB0 or None to not user port")

    args = parser.parse_args()

    log.info('opening UDP port {} to receive frames from producer'.format(PORT))
    socket.setdefaulttimeout(1) # set timeout to allow keyboard commands to cv window
    server_socket: socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    address = ("", PORT)
    server_socket.bind(address)
    load_latest_model_convert_to_tflite()
    interpreter, input_details, output_details=load_tflite_model(MODEL_DIR)


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

    if len(tf.config.list_physical_devices('GPU')) > 0:
        log.info('GPU is available')
    else:
        log.warning('GPU not available (check tensorflow/cuda setup)')


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

    # map from prediction of symbol to correct hand command to beat human
    # prediction symbol ('background' 'rock','scissors', 'paper'), class number (0-3)
    # see Arduino firmware https://github.com/SensorsINI/Dextra-robot-hand-firmware for the commands and hand symbols shown
    pred_to_cmd_dict={0:b'2',1:b'3',2:b'1',3:None}
    cmd_voter = majority_vote(window_length=5, num_classes=4)

    cv2.namedWindow('RoshamboCNN', cv2.WINDOW_NORMAL)
    if not cv2_resized:
        cv2.resizeWindow('RoshamboCNN', 600, 600)

    log.info('in display, hit x to exit or spacebar to show demo movement')
    while True:
        timestr = time.strftime("%Y%m%d-%H%M")
        # with Timer('overall consumer loop', numpy_file=f'{DATA_FOLDER}/consumer-frame-rate-{timestr}.npy', show_hist=True):
        with Timer('overall consumer loop', numpy_file=None, show_hist=False):
            with Timer('recieve UDP'):
                try:
                    receive_data = server_socket.recv(UDP_BUFFER_SIZE)
                except socket.timeout:
                    log.debug('timeout for frame from DVS')
                    k = cv2.waitKey(1) & 0xFF # 1ms poll
                    if k==ord('x'):
                        break
                    elif k==ord(' '):
                        show_demo_sequence()
                    else:
                        maybe_show_demo_sequence()
                    continue


            with Timer('unpickle and normalize/reshape'):
                (frame_number,timestamp, img) = pickle.loads(receive_data)
                dropped_frames=frame_number-last_frame_number-1
                if dropped_frames>0:
                    log.warning(f'Dropped {dropped_frames} frames from producer')
                last_frame_number=frame_number
                # img = (1./255)*np.reshape(img, [IMSIZE, IMSIZE,1])
            with Timer('run CNN', numpy_file=None, show_hist=SHOW_STATISTICS_AT_END):
                # pred = model.predict(img[None, :])
                pred_class_name, pred_idx, pred_vector=classify_img(img, interpreter, input_details, output_details)

            if pred_idx<=3: # symbol recognized (or background==3)
                cmd=pred_to_cmd_dict[pred_idx] # start with no command sent to hand
                if USE_MAJORITY_VOTE:
                    vote = cmd_voter.new_prediction_and_vote(pred_idx)
                    if not vote is None: 
                        cmd = pred_to_cmd_dict[vote]
                        pred_class_name=list(CLASS_DICT.keys())[list(CLASS_DICT.values()).index(vote)]

                # now send a command if there is one and we have not sent too recently
                if not serial_port is None and not cmd is None and time.time()-time_last_sent_cmd>MIN_INTERVAL_S_BETWEEN_CMDS:
                    try:
                        arduino_serial_port.write(cmd)
                        time_last_sent_cmd=time.time()
                        log.debug(f'sent cmd {cmd} for pred_idx {pred_idx} and detected symbol {pred_class_name}')

                    except serial.serialutil.SerialException as e:
                        log.error(f'Error writing to serial port {SERIAL_PORT} with cmd {cmd} for detected symbol {pred_class_name}: {e}')



            cv2.namedWindow('RoshamboCNN', cv2.WINDOW_NORMAL)
            cv2.putText(img, pred_class_name, (1, 10), cv2.FONT_HERSHEY_PLAIN, .7, (255, 255, 255), 1)
            cv2.imshow('RoshamboCNN', 1 - img.astype('float') / 255)
            if not cv2_resized:
                cv2.resizeWindow('RoshamboCNN', 600, 600)
                cv2_resized = True
            k = cv2.waitKey(1) & 0xFF # 1ms poll
            if k==ord('x'):
                break
            elif k == ord('p'):
                print_timing_info()
            elif k==ord(' '):
                show_demo_sequence()

            maybe_show_demo_sequence()

            # save time since frame sent from producer
            dt=time.time()-timestamp
            with Timer('producer->consumer inference delay',delay=dt, show_hist=False):
                pass

if __name__ == '__main__':
    consumer(queue=None)

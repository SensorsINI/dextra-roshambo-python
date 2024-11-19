""" Shared stuff between producer and consumer
 Author: Tobi Delbruck
 """
import math
import os
import sys
import time
from subprocess import TimeoutExpired

import cv2
import numpy as np
import atexit
from engineering_notation import EngNumber  as eng  # only from pip
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime
from my_logger import my_logger
log=my_logger(__name__)

PORT = 12000  # UDP port used to send frames from producer to consumer
IMSIZE = 64  # input image size, must match model
UDP_BUFFER_SIZE = int(math.pow(2, math.ceil(math.log(IMSIZE * IMSIZE + 1000) / math.log(2))))

EVENT_COUNT_CLIP_VALUE = 16  # full count value for colleting histograms of DVS events
SHOW_DVS_OUTPUT = False # producer shows the accumulated DVS frames as aid for focus and alignment
MIN_PRODUCER_FRAME_INTERVAL_MS=5.0 # inference takes about 3ms and normalization takes 1ms, hence at least 2ms
        # limit rate that we send frames to about what the GPU can manage for inference time
        # after we collect sufficient events, we don't bother to normalize and send them unless this time has
        # passed since last frame was sent. That way, we make sure not to flood the consumer
MAX_SHOWN_DVS_FRAME_RATE_HZ=15 # limits cv2 rendering of DVS frames to reduce loop latency for the producer
FULLSCREEN=True # for kiosk demo autostart

DATA_FOLDER = 'data' #'data'  # new samples stored here
SERIAL_PORT = "/dev/ttyUSB0"  # port to talk to arduino finger controller

SRC_DATA_FOLDER = '/home/tobi/Downloads/dextra_roshambo/source_data'
TRAIN_DATA_FOLDER='/home/tobi/Downloads/dextra_roshambo/training_dataset' # the actual training data that is produced by split from dataset_utils/make_train_valid_test()


MODEL_DIR='model' # where models stored
MODEL_BASE_NAME= 'model_185' # base name of checkpoint .index and .data files in the MODEL_DIR
DEXTRA_NET_BASE_NAME= 'dextra_roshambo' # base name
TFLITE_FILE_NAME= DEXTRA_NET_BASE_NAME + '.tflite' # tflite model is stored in same folder as full-blown TF2 model
SYMBOL_TO_PRED_DICT={'background':3, 'paper':0, 'scissors': 1, 'rock':2} # maps from symbol name to prediction number from CNN
PRED_TO_SYMBOL_DICT={v: k for k, v in SYMBOL_TO_PRED_DICT.items()} # maps from prediction number to symbol name
MIN_INTERVAL_S_BETWEEN_CMDS=3e-3 # use to limit serial port command rate
PREDICTION_VOTING_METHOD= 'sequence' # 'majority', None #  base hand movements on majority vote over past few predictions of human hand symbol to improve accuracy (but increase latency)

SHOW_STATISTICS_AT_END=False # set True to show timing histograms

# museum settings
MUSEUM_OPENING_TIME=datetime.strptime('11:50','%H:%M').time() # start when attracting movements are shown
MUSEUM_CLOSING_TIME=datetime.strptime('17:15','%H:%M').time()
MUSEUM_DEMO_MOVEMENT_INTERVAL_M=3  # minutes between showing attracting movements of RSP movement if no cmd has been sent
MUSEUM_LOGGING_FILE="actions-log" # # logging data to track activity, this is basename, actual name is e.g. file-YYYYMMDD-hhmm.csv
MUSEUM_ACTIONS_LOGGING_INTERVAL_MINUTES=10 # minutes between logging number of hand actions
MUSEUM_ACTIONS_CSV_LOG_FILE_CREATION_INTERVAL_HOURS=24 # how many hours between creating new activity CSV file
MUSEUM_I_AM_ALIVE_LOG_INTERVAL_MINUTES=10 # how many minutes between logging "I'm alive" messages
MUSEUM_SCAN_FOR_RESTART_FILE=True # periodically check if there is a file named "RESTART" and restart myself if found, deleting the file first.
MUSEUM_SLEEP_TIME_LOCAL="18:30" # local time every day to sleep computer, used by schedule to run the sleep command
MUSEUM_WAKE_TIME_UTC="07:45" # time that computer wakes in UTC time (time of computer RTC clock, checked in bios or by timedatectl)
MUSEUM_SCREEN_DIM_NO_ACTIONS_TIMEOUT_S=10 # time to dim screen if no actions for this many seconds

# saving frames 
SAVE_FRAMES_INTERVAL=10
SAVE_FRAMES_STORAGE_LOCATION='frames' # saved in logging folder too, in this subfolder
SAVE_FRAMES_DISK_FREE_STOP_LIMIT_GB=100

import signal
def alarm_handler(signum, frame):
    raise TimeoutError
def input_with_timeout(prompt, timeout=30):
    """ get input with timeout

    :param prompt: the prompt to print
    :param timeout: timeout in seconds, or None to disable

    :returns: the input
    :raises: TimeoutError if times out
    """
    # set signal handler
    if timeout is not None:
        signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(timeout) # produce SIGALRM in `timeout` seconds
    try:
        time.sleep(.5) # get input to be printed after logging
        return input(prompt)
    except TimeoutError as to:
        raise to
    finally:
        if timeout is not None:
            signal.alarm(0) # cancel alarm

def yes_or_no(question, default='y', timeout=None):
    """ Get y/n answer with default choice and optional timeout

    :param question: prompt
    :param default: the default choice, i.e. 'y' or 'n'
    :param timeout: the timeout in seconds, default is None

    :returns: True or False
    """
    if default is not None and (default!='y' and default!='n'):
        log.error(f'bad option for default: {default}')
        quit(1)
    y='Y' if default=='y' else 'y'
    n='N' if default=='n' else 'n'
    while "the answer is invalid":
        try:
            to_str='' if timeout is None else f'(Timeout {default} in {timeout}s)'
            reply = str(input_with_timeout(f'{question} {to_str} ({y}/{n}): ',timeout=timeout)).lower().strip()
        except TimeoutError:
            log.warning(f'timeout expired, returning default={default} answer')
            reply=''
        if len(reply)==0 or reply=='':
            return True if default=='y' else False
        elif reply[0].lower() == 'y':
            return True
        if reply[0].lower() == 'n':
            return False

timers = {}
times = {}
class Timer:
    def __init__(self, timer_name='', delay=None, show_hist=False, numpy_file=None):
        """ Make a Timer() in a _with_ statement for a block of code.
        The timer is started when the block is entered and stopped when exited.
        The Timer _must_ be used in a with statement.

        :param timer_name: the str by which this timer is repeatedly called and which it is named when summary is printed on exit
        :param delay: set this to a value to simply accumulate this externally determined interval
        :param show_hist: whether to plot a histogram with pyplot
        :param numpy_file: optional numpy file path
        """
        self.timer_name = timer_name
        self.show_hist = show_hist
        self.numpy_file = numpy_file
        self.delay=delay

        if self.timer_name not in timers.keys():
            timers[self.timer_name] = self
        if self.timer_name not in times.keys():
            times[self.timer_name]=[]

    def __enter__(self):
        if self.delay is None:
            self.start = time.time()
        return self

    def __exit__(self, *args):
        if self.delay is None:
            self.end = time.time()
            self.interval = self.end - self.start  # measured in seconds
        else:
            self.interval=self.delay
        times[self.timer_name].append(self.interval)

    def print_timing_info(self,stream=None):
        a = np.array(times[self.timer_name])
        if len(a)==0:
            log.error(f'Timer {self.timer_name} has no statistics; was it used without a with statement?')
            return
        timing_mean = np.mean(a) # todo use built in print method for timer
        timing_std = np.std(a)
        timing_median = np.median(a)
        timing_min = np.min(a)
        timing_max = np.max(a)
        log.info('{} n={}: {}s +/- {}s (median {}s, min {}s max {}s)'.format(self.timer_name, len(a),
                                                                          eng(timing_mean), eng(timing_std),
                                                                          eng(timing_median), eng(timing_min),
                                                                          eng(timing_max)))

def print_timing_info():
    print('== Timing statistics ==')
    for k,v in times.items():  # k is the name, v is the list of times
        a = np.array(v)
        timing_mean = np.mean(a)
        timing_std = np.std(a)
        timing_median = np.median(a)
        timing_min = np.min(a)
        timing_max = np.max(a)
        log.info('\n{} n={}: {}s +/- {}s (median {}s, min {}s max {}s)'.format(k, len(a),
                                                                          eng(timing_mean), eng(timing_std),
                                                                          eng(timing_median), eng(timing_min),
                                                                          eng(timing_max)))
        if timers[k].numpy_file is not None:
            try:
                log.info(f'saving timing data for {k} in numpy file {timers[k].numpy_file}')
                log.info('there are {} times'.format(len(a)))
                np.save(timers[k].numpy_file, a)
            except Exception as e:
                log.error(f'could not save numpy file {timers[k].numpy_file}; caught {e}')

        if timers[k].show_hist:

            def plot_loghist(x, bins):
                hist, bins = np.histogram(x, bins=bins)
                logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
                plt.hist(x, bins=logbins)
                plt.xscale('log')

            dt = np.clip(a,1e-6, None)
            # logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
            plot_loghist(dt,bins=100)
            plt.xlabel('interval[ms]')
            plt.ylabel('frequency')
            plt.title(k)
            plt.show()

# this will print all the timer values upon termination of any program that imported this file
atexit.register(print_timing_info)

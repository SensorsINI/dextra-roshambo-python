"""
producer of DVS frames for classification of hand symbols by consumer processs
Authors: Tobi Delbruck Nov 2020, Oct 2024
"""

import atexit
import pickle
from pathlib import Path
from typing import Optional, Union

import cv2
import sys
import math
import time
import numpy.ma as ma
import socket
import numpy as np
from globals_and_utils import *
from engineering_notation import EngNumber  as eng # only from pip
import argparse
import multiprocessing.connection as mpc
from multiprocessing import  Pipe,Queue

from pyaer.davis import DAVIS
from pyaer.dvs128 import DVS128

from pyaer import libcaer
from my_logger import my_logger
log=my_logger(__name__)

# camera type is automatically detected in open_camera()
CAMERA_TYPES=[DVS128,DAVIS]
CAMERA_TO_BIASES_DICT={'DVS128':'./configs/dvs128_config.json', 'DAVIS': './configs/davis346_config.json'}
CAMERA_UNPLUGGED_TIMEOUT_S=5 # how long with zero events to give up on camera and close/repopen (in case of unplugged or wakeup from sleep)




def producer(queue:Queue):
    """ produce frames for consumer

    :param queue: possible Queue to send to instead of UDP socket
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_address = ('', PORT)

    EVENT_COUNT_PER_FRAME = 1500


    record=None
    spacebar_records=True
    log.info(f'recording to {record} with spacebar_records={spacebar_records}')

    recording_folder = None
    if record is not None:
        recording_folder = os.path.join(DATA_FOLDER, 'recordings', record)
        log.info(f'recording frames to {recording_folder}')
        Path(recording_folder).mkdir(parents=True, exist_ok=True)

    def open_camera()->Optional[Union[DVS128,DAVIS]]:
        """ Opens the DAVIS camera, set biases, and returns device handle to it
        :return: device handle
        """
        nonlocal EVENT_COUNT_PER_FRAME
        dvs=None
        for c in CAMERA_TYPES:
            try:
                dvs=c(noise_filter=True) # open the camera
                log.info(f'opened camera {dvs}')
                break
            except Exception as e:
                log.info(f'cannot open camera type {c}: {e}')
        if dvs is None:
            log.warning(f"could not open any camera type in {CAMERA_TYPES}")
            return None
        
        EVENT_COUNT_PER_FRAME=None
        if type(dvs) is DVS128:
            EVENT_COUNT_PER_FRAME=1500 
        elif type(dvs) is DAVIS:
            EVENT_COUNT_PER_FRAME=5000
        else:
            raise Exception(f'{dvs} type is unknown')
        log.info(f'set accumulated event count/frame to {EVENT_COUNT_PER_FRAME} for camera {dvs}')

        print("DVS USB ID:", dvs.device_id)
        if dvs.device_is_master:
            print("DVS is master.")
        else:
            print("DVS is slave.")
        print("DVS Serial Number:", dvs.device_serial_number)
        print("DVS String:", dvs.device_string)
        print("DVS USB bus Number:", dvs.device_usb_bus_number)
        print("DVS USB device address:", dvs.device_usb_device_address)
        print("DVS size X:", dvs.dvs_size_X)
        print("DVS size Y:", dvs.dvs_size_Y)
        try:
            print("Logic Version:", dvs.logic_version)
        except:
            print("DVS128 camera has no logic version information available")
        try:
            print("Background Activity Filter:",
                dvs.dvs_has_background_activity_filter)
        except:
            print("DVS128 has no background activity denoising filter")
        try:
            print("Color Filter", dvs.aps_color_filter, type(dvs.aps_color_filter))
            print(dvs.aps_color_filter == 1)
        except:
            print("DVS128 has no color filter setting to print or set")

        # device.start_data_stream()
        assert (dvs.send_default_config())
        # attempt to set up USB host buffers for acquisition thread to minimize latency
        assert (dvs.set_config(
            libcaer.CAER_HOST_CONFIG_USB,
            libcaer.CAER_HOST_CONFIG_USB_BUFFER_NUMBER,
            8))
        assert (dvs.set_config(
            libcaer.CAER_HOST_CONFIG_USB,
            libcaer.CAER_HOST_CONFIG_USB_BUFFER_SIZE,
            4096))
        assert (dvs.data_start())
        assert (dvs.set_config(
            libcaer.CAER_HOST_CONFIG_PACKETS,
            libcaer.CAER_HOST_CONFIG_PACKETS_MAX_CONTAINER_INTERVAL,
            1000)) # set max interval to this value in us. Set to not produce too many packets/sec here, not sure about reasoning
        assert (dvs.set_data_exchange_blocking())

        # setting bias after data stream started
        bias_json_file=CAMERA_TO_BIASES_DICT[type(dvs).__name__]
        log.info(f'setting biases from {bias_json_file}')
        dvs.set_bias_from_json(bias_json_file)
        return dvs

    def write_next_image(dir:str, idx:int, img):
        """ Saves data sample image

        :param dir: the folder
        :param idx: the current index number
        :param img: the image to save, which should be monochrome uint8 and which is saved as default png format
        :returns: the next index
        """
        while True:
            n=f'{dir}/{idx:04d}.png'
            if not os.path.isfile(n):
                break
            idx+=1
        try:
            cv2.imwrite(n, img)
        except Exception as e:
            log.error(f'error saving {n}: caught {e}')
        return idx

    dvs=None

    histrange = [(0, v) for v in (IMSIZE, IMSIZE)]  # allocate DVS frame histogram to desired output size
    npix = IMSIZE * IMSIZE
    cv2_resized = False
    last_cv2_frame_time = time.time()
    frame=None
    frame_number=0
    recording_frame_number = 0
    time_last_frame_sent=time.time()
    frames_dropped_counter=0
    save_next_frame=not spacebar_records # if we don't supply the option, it will be False and we want to then save all frames
    last_events_recieved_time=time.time()

    def cleanup():
        if dvs:
            log.info('closing {}'.format(dvs))
            dvs.shutdown()
        cv2.destroyAllWindows()

    atexit.register(cleanup)

    try:
        numpy_file = None # TODO uncomment to save data f'{DATA_FOLDER}/producer-frame-rate-{timestr}.npy'
        log.info('*********** starting main producer loop')
        while True:

            if dvs is None:
                dvs=open_camera()
                if dvs is None:
                    log.warning('no DVS camera found, sleeping to next try to open one')
                    time.sleep(1)
                    continue
                log.info(f'camera {dvs} successfully opened')

            with Timer('overall producer frame rate', numpy_file=numpy_file , show_hist=SHOW_STATISTICS_AT_END) as timer_overall:
                with Timer('accumulate DVS'):
                    events = None
                    while (not dvs is None) and (events is None or ((not events is None) and len(events)<EVENT_COUNT_PER_FRAME)):
                        # pol_events, num_pol_event,_, _, _, _, _, _ = dvs.get_event()
                        # dvs.get_event() might return None if camera is unplugged
                        # pol_events, num_pol_event, *_ = dvs.get_event() # ignore other return values since only brightness change events are used from DVS and DAVIS
                        ret = dvs.get_event() # ignore other return values since only brightness change events are used from DVS and DAVIS
                        if not ret is None:
                            pol_events, num_pol_event, *_ = ret
                        else:
                            num_pol_event=0
                        # assemble 'frame' of EVENT_COUNT events
                        if  num_pol_event>0:
                            last_events_recieved_time=time.time()
                            if events is None:
                                events=pol_events
                            else:
                                events = np.vstack([events, pol_events]) # otherwise tack new events to end
                        else: # no events this call
                            if time.time()-last_events_recieved_time>CAMERA_UNPLUGGED_TIMEOUT_S:
                                log.error(f'time since last recieved any events is >{CAMERA_UNPLUGGED_TIMEOUT_S}s, will try to reopen camera')
                                try:
                                    dvs.shutdown()
                                except:
                                    pass
                                dvs=None
                if dvs is None:
                    continue
                # log.debug('got {} events (total so far {}/{} events)'
                #          .format(num_pol_event, 0 if events is None else len(events), EVENT_COUNT))
                dtMs = (time.time() - time_last_frame_sent)*1e3
                if recording_folder is None and dtMs<MIN_PRODUCER_FRAME_INTERVAL_MS:
                    log.debug(f'frame #{frames_dropped_counter} after only {dtMs:.3f}ms, discarding to collect newer frame')
                    frames_dropped_counter+=1
                    continue # just collect another frame since it will be more timely

                log.debug(f'after dropping {frames_dropped_counter} frames, got one after {dtMs:.1f}ms')
                frames_dropped_counter=0
                with Timer('normalization'):
                    # if frame is None: # debug timing
                        # take DVS coordinates and scale x and y to output frame dimensions using flooring math
                        xfac = float(IMSIZE) / dvs.dvs_size_X
                        yfac = float(IMSIZE) / dvs.dvs_size_Y
 
                        events[:,1]=np.floor(events[:,1]*xfac)
                        events[:,2]=np.floor(events[:,2]*yfac)
                        frame, _, _ = np.histogram2d(events[:, 2], events[:, 1], bins=(IMSIZE, IMSIZE), range=histrange)
                        fmax_count=np.max(frame)
                        frame[frame > EVENT_COUNT_CLIP_VALUE]=EVENT_COUNT_CLIP_VALUE
                        frame= (255. / EVENT_COUNT_CLIP_VALUE) * frame # max pixel will have value 255

                # statistics
                focc=np.count_nonzero(frame)
                frame=frame.astype('uint8')
                log.debug('from {} events, frame has occupancy {}% max_count {:.1f} events'.format(len(events), eng((100.*focc)/npix), fmax_count))

                with Timer('send frame'):
                    time_last_frame_sent=time.time()
                    data = pickle.dumps((frame_number, time_last_frame_sent, frame)) # send frame_number to allow determining dropped frames in consumer
                    frame_number+=1
                    client_socket.sendto(data, udp_address)
                if recording_folder is not None and save_next_frame:
                    recording_frame_number=write_next_image(recording_folder,recording_frame_number,frame)
                    print('.',end='')
                    if recording_frame_number%80==0:
                        print('')
                if SHOW_DVS_OUTPUT:
                    t=time.time()
                    if t-last_cv2_frame_time>1./MAX_SHOWN_DVS_FRAME_RATE_HZ:
                        last_cv2_frame_time=t
                        with Timer('show DVS image'):
                            # min = np.min(frame)
                            # img = ((frame - min) / (np.max(frame) - min))
                            cv2.namedWindow('DVS', cv2.WINDOW_NORMAL)
                            cv2.imshow('DVS', 1-frame.astype('float')/255)
                            if not cv2_resized:
                                cv2.resizeWindow('DVS', 600, 600)
                                cv2_resized = True
                            k=    cv2.waitKey(1) & 0xFF
                            if k== ord('q'):
                                if recording_folder is not None:
                                    log.info(f'*** recordings of {recording_frame_number - 1} frames saved in {recording_folder}')
                                break
                            elif k==ord('p'):
                                print_timing_info()
                            elif spacebar_records and k==ord(' '):
                                save_next_frame=True
                            else:
                                save_next_frame=not spacebar_records

    except KeyboardInterrupt:
        dvs.shutdown()
        if recording_folder is not None:
            log.info(f'*** recordings of {recording_frame_number-1} frames saved in {recording_folder}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='producer: Generates DVS frames for roshambo to process in consumer', allow_abbrev=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--record", type=str, default=None,
        help="record DVS frames into folder DATA_FOLDER/collected/<name>")
    parser.add_argument(
        "--spacebar_records", action='store_true',
        help="only record when spacebar pressed down")
    args = parser.parse_args()

    producer(queue=None)

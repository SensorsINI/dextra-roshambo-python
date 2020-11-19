"""
producer of DVS frames for classification of joker/nonjoker by consumer processs
Authors: Tobi Delbruck, Yuhuang Hu, Shasha Guo,  Min Liu Oct 2020
"""

import atexit
import pickle
from pathlib import Path

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

from pyaer.davis import DAVIS
from pyaer import libcaer

log=my_logger(__name__)



def producer(args):
    """ produce frames for consumer

    :param record: record frames to a folder name record
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_address = ('', PORT)

    device = DAVIS(noise_filter=True)

    def cleanup():
        log.info('closing {}'.format(device))
        device.shutdown()
        cv2.destroyAllWindows()

    atexit.register(cleanup)
    record=args.record
    spacebar_records=args.spacebar_records
    log.info(f'recording to {record} with spacebar_records={spacebar_records}')

    recording_folder = None
    if record is not None:
        recording_folder = os.path.join(DATA_FOLDER, 'recordings', record)
        log.info(f'recording frames to {recording_folder}')
        Path(recording_folder).mkdir(parents=True, exist_ok=True)

    print("DVS USB ID:", device.device_id)
    if device.device_is_master:
        print("DVS is master.")
    else:
        print("DVS is slave.")
    print("DVS Serial Number:", device.device_serial_number)
    print("DVS String:", device.device_string)
    print("DVS USB bus Number:", device.device_usb_bus_number)
    print("DVS USB device address:", device.device_usb_device_address)
    print("DVS size X:", device.dvs_size_X)
    print("DVS size Y:", device.dvs_size_Y)
    print("Logic Version:", device.logic_version)
    print("Background Activity Filter:",
          device.dvs_has_background_activity_filter)
    print("Color Filter", device.aps_color_filter, type(device.aps_color_filter))
    print(device.aps_color_filter == 1)

    # device.start_data_stream()
    assert (device.send_default_config())
    # attempt to set up USB host buffers for acquisition thread to minimize latency
    assert (device.set_config(
        libcaer.CAER_HOST_CONFIG_USB,
        libcaer.CAER_HOST_CONFIG_USB_BUFFER_NUMBER,
        8))
    assert (device.set_config(
        libcaer.CAER_HOST_CONFIG_USB,
        libcaer.CAER_HOST_CONFIG_USB_BUFFER_SIZE,
        4096))
    assert (device.data_start())
    assert (device.set_config(
        libcaer.CAER_HOST_CONFIG_PACKETS,
        libcaer.CAER_HOST_CONFIG_PACKETS_MAX_CONTAINER_INTERVAL,
        4000)) # set max interval to this value in us. Set to not produce too many packets/sec here, not sure about reasoning
    assert (device.set_data_exchange_blocking())

    # setting bias after data stream started
    device.set_bias_from_json("./configs/davis346_config.json")
    xfac = float(IMSIZE) / device.dvs_size_X
    yfac = float(IMSIZE) / device.dvs_size_Y
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

    try:
        timestr = time.strftime("%Y%m%d-%H%M")
        numpy_file = f'{DATA_FOLDER}/producer-frame-rate-{timestr}.npy'
        while True:

            with Timer('overall producer frame rate', numpy_file=numpy_file , show_hist=True) as timer_overall:
                with Timer('accumulate DVS'):
                    events = None
                    while events is None or len(events)<EVENT_COUNT_PER_FRAME:
                        pol_events, num_pol_event,_, _, _, _, _, _ = device.get_event()
                        # assemble 'frame' of EVENT_COUNT events
                        if  num_pol_event>0:
                            if events is None:
                                events=pol_events
                            else:
                                events = np.vstack([events, pol_events]) # otherwise tack new events to end
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
        device.shutdown()
        if recording_folder is not None:
            log.info(f'*** recordings of {recording_frame_number-1} frames saved in {recording_folder}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='producer: Generates DVS frames for trixy to process in consumer', allow_abbrev=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--record", type=str, default=None,
        help="record DVS frames into folder DATA_FOLDER/collected/<name>")
    parser.add_argument(
        "--spacebar_records", action='store_true',
        help="only record when spacebar pressed down")
    args = parser.parse_args()

    try:
        producer(args)
    except Exception as e:
        log.error(f'Error: {e}')
        sys.exit()

# runs entire PDAVIS E2P demo
# author: Tobi Delbruck 2023

import multiprocessing as mp
import time
from multiprocessing import Process, Pipe, Queue, SimpleQueue

from producer import producer
from consumer import consumer
from utils.get_logger import get_logger
from utils.kbhit import KBHit
from globals_and_utils import *
from pathlib import Path

log = get_logger(__name__)

def main():
    kb=None
    try:
        kb = KBHit()  # can only use in posix terminal; cannot use from spyder ipython console for example
        kbAvailable = True
    except:
        kbAvailable = False

    def print_help():
        print('x or ESC:  exit')
        # print('space: pause/unpause')
        # print('r toggle recording')

    mp.set_start_method('spawn') # https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    queue=Queue()
    # parent_conn, child_conn = Pipe() # half duplex, only send and recieve on other side
    con, pro = start_processes(queue)
    if kbAvailable:
        print_help()
    while True:
        # log.debug('waiting for consumer and producer processes to join')
        if kbAvailable and kb.kbhit():
            print('.',end='')
            ch = kb.getch()
            if ch == 'h' or ch == '?':
                print_help()
            elif ord(ch) == 27 or ch == 'x':  # ESC, 'x'
                log.info('x or ESC typed, stopping producer and consumer')
                stop_processes(con, pro)
                break
        if not con.is_alive() or not pro.is_alive():
            log.warning('either or both producer or consumer process(es) ended, terminating pdavis_demo loop')
            if pro.is_alive():
                log.debug('terminating producer')
                pro.terminate()
            if con.is_alive():
                log.debug('terminating consumer')
                con.terminate()
            break
        if MUSEUM_SCAN_FOR_RESTART_FILE:
            restart_path=Path("RESTART")
            try:
                if restart_path.is_file():
                    log.info('"RESTART" file found, restarting producer and consumer processes')
                    restart_path.unlink()
                    stop_processes(consumer=con,producer=pro)
                    pro.join()
                    con.join()
                    con,pro=start_processes(queue=queue)
            except Exception as e:
                log.error(f'could not restart: {e}')

        time.sleep(5)
    log.debug('joining producer and consumer processes')
    pro.join()
    con.join()
    log.debug('both consumer and producer processes have joined, done')

def stop_processes(consumer, producer):
    log.info("\nterminating producer and consumer....")
    consumer.terminate()
    producer.terminate()

def start_processes(queue):
    log.debug('starting Roshambo demo consumer process')
    con = Process(target=consumer, args=(queue,),name='consumer')
    con.start()
    log.debug('sleeping 5 seconds')
    time.sleep(5) # give some time to load DNN
    log.debug('starting Roshambo demo producer process')
    pro = Process(target=producer, args=(queue,),name='producer')
    pro.start()
    return con,pro
    # quit(0)

if __name__ == '__main__':
    main()


# runs entire PDAVIS E2P demo
# author: Tobi Delbruck 2023

import multiprocessing as mp
import subprocess
import time
from multiprocessing import Process, Pipe, Queue

from producer import producer
from consumer import consumer
from utils.get_logger import get_logger
from utils.kbhit import KBHit
from globals_and_utils import *
from pathlib import Path
import schedule

log=my_logger('roshambo')

TEST_SLEEP=False # debug/test sleep/wake with 2 minute intervals

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
    log.info(f'scheduling sleep every day at {MUSEUM_SLEEP_TIME_LOCAL} UTC and wake at {MUSEUM_WAKE_TIME_UTC}')
    if TEST_SLEEP:
        schedule.every(2).minutes.do(sleep_till_tomorrow) # debug
    else:
        schedule.every().day.at(MUSEUM_SLEEP_TIME_LOCAL).do(sleep_till_tomorrow)

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
            log.info('either or both producer or consumer process(es) ended, terminating loop')
            if pro.is_alive():
                log.debug('terminating producer')
                pro.terminate()
            if con.is_alive():
                log.debug('terminating consumer')
                con.terminate()
            break
        schedule.run_pending()
        time.sleep(5)
        # end of main dextra loop
    log.debug('joining producer and consumer processes')
    pro.join()
    con.join()
    log.debug('both consumer and producer processes have joined, done')

def stop_processes(consumer, producer):
    log.info("\nterminating producer and consumer....")
    consumer.terminate()
    producer.terminate()


def sleep_till_tomorrow():
    if TEST_SLEEP:
        log.info('testing sleep  with 1 minute sleep')
        args=f' -u -m mem -t $(date -d "+ 1 minute" +%s)'
    else:
        args=f' -u -m mem -t $(date -d "tomorrow {MUSEUM_WAKE_TIME_UTC}" +%s)' # -u assume hardware clock set to UTC, -m mem suspend to RAM, -d 
    cmd=f'/usr/sbin/rtcwake {args}'
    log.info(f'****** going to sleep in 5s with "{cmd}"')
    time.sleep(5)
    result=None
    result=subprocess.run(cmd,check=False, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,text=True)
    if result.returncode==0:
        log.info(f'rtcwake call successful: result={result}')
    else:
        log.error(f'could not execute "{cmd}":\n result={result}\n')
    time.sleep(3)
    log.info('****** woke from sleep')

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


"""
Customized logger
Authors: Tobi Delbruck, Nov 2020, Oct 2024
"""
import logging
from logging.handlers import TimedRotatingFileHandler
from multiprocessing import Process, current_process
import os
from pathlib import Path
import platform
import time # hostname for unique log file name

LOG_DIR='logging' # where all logging (python logging and saved frames) are stored
LOGGING_LEVEL = logging.INFO
LOG_FILE='dextra' # base name of rotating log file; see my_logger.py
LOG_ROTATION_INTERVAL_HOURS=24

class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# https://stackoverflow.com/questions/44691558/suppress-multiple-messages-with-same-content-in-python-logging-module-aka-log-co
class DuplicateFilter(logging.Filter):

    def __init__(self, name = ""):
        super().__init__(name)
        self.repeat_count=0
        self.last_log=None

    def filter(self, record):
        # add other fields if you need more granular comparison, depends on your app
        current_log = (record.module, record.levelno, record.msg)
        # repeated messages are based on module, level and msg (not counting the timestamp of the log record)
        if current_log != self.last_log:
            if self.repeat_count>0:
                record.msg=f'(suppressed {self.repeat_count} repeated messages) '+record.msg
            self.last_log = current_log
            self.repeat_count=0
            return True
        else:
            self.repeat_count+=1
        return False

# https://stackoverflow.com/questions/58300443/creating-log-files-with-loggings-timedrotatingfilehandler
class MyCustomTimedRotatingFileHandler(TimedRotatingFileHandler):
    def namer(self, default_name):
        full_name = Path(default_name)
        file_name = full_name.name
        directory = full_name.parent

        first, middle, last = file_name.partition(".log")
        new_file_name = f"{first}{last}{middle}"
        full_name= directory / new_file_name
        # print(f'log file {full_name}')
        return full_name


def my_logger(name):
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    log = logging.getLogger(name)
    log.setLevel(LOGGING_LEVEL)
    # create console handler
    console_handler = logging.StreamHandler()
    
    console_handler.setFormatter(CustomFormatter())
    log.addHandler(console_handler)
    log.addFilter(DuplicateFilter())
    if LOG_FILE:
        fn=LOG_FILE+'-'+platform.node()+'-'+name+".log"
        path=os.path.join(LOG_DIR,fn)
        log.info(f'adding TimedRotatingFileHandler for logging output to {path} rotated every {LOG_ROTATION_INTERVAL_HOURS}h')
        # fh = logging.handlers.TimedRotatingFileHandler(path,when="H",
        #                                                interval=MUSEUM_ACTIONS_CSV_LOG_FILE_CREATION_INTERVAL_HOURS,
        #                                                backupCount=100) 
        fh = MyCustomTimedRotatingFileHandler(path,when="H",
                                                       interval=LOG_ROTATION_INTERVAL_HOURS,
                                                       backupCount=7)
        fh = MyCustomTimedRotatingFileHandler(path,when="S",
                                                       interval=5,
                                                       backupCount=3)
        fh.setFormatter(CustomFormatter())
        log.addHandler(fh)
    return log

class my_process(Process):
        def __init__(self):
             Process.__init__(self)

        def run(self):
            log=my_logger(self.name)
            log.info('first log')
            for i in range(1000):
                    log.info(f'2nd log, repeat')
                    time.sleep(.1)
            log.info('3rd message')


if __name__ == '__main__':

    a = Process(target=my_process,name='a')
    b = Process(target=my_process,name='b')
    a.start()
    b.start()
    log=my_logger('main')
    log.info('first log')
    for i in range(100):
            log.info(f'2nd log, repeat')
            time.sleep(.1)

    log.info('3rd message')
    a.join()
    b.join()


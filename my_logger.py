"""
Customized logger
Authors: Tobi Delbruck, Nov 2020, Oct 2024
"""
import logging

LOGGING_LEVEL = logging.INFO

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
                record.msg=f'(suppressed {self.repeat_count} repeated messaages) '+record.msg
            self.last_log = current_log
            self.repeat_count=0
            return True
        else:
            self.repeat_count+=1
        return False
 
def my_logger(name):
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(LOGGING_LEVEL)
    # create console handler
    console_handler = logging.StreamHandler()
    
    console_handler.setFormatter(CustomFormatter())
    logger.addHandler(console_handler)
    logger.addFilter(DuplicateFilter())
    return logger

   

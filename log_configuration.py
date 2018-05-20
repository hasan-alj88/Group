import functools
import logging
import sys
import time
from logging.handlers import RotatingFileHandler

default_timer = time.clock if (sys.platform == "win32") else time.time


def create_logger():
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    # Handlers..
    # Stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.ERROR)
    # log file handler
    file_handler = RotatingFileHandler("logs\\GroupPythonProject.log", maxBytes=2 ** 20, backupCount=100)
    file_handler.setLevel(logging.DEBUG)
    # create formatter
    log_format = logging.Formatter('%(asctime)s\t||\t%(message)s')
    # add formatter to handlers
    stream_handler.setFormatter(log_format)
    file_handler.setFormatter(log_format)
    # add handles to logger
    # log.addHandler(stream_handler)
    log.addHandler(file_handler)
    return log


logger = create_logger()


def log_decorator(defined_logger):
    """
    A decorator that wraps the passed in function and logs
    exceptions should one occur
    @param defined_logger: The logging object
    """
    def decorator(func):
        @functools.wraps(func)
        def log_wrapper(*args, **kwargs):
            try:
                defined_logger.debug('{}({}) function will start.'.format(func.__qualname__, args))
                timer_start = default_timer()
                ret = func(*args, **kwargs)
                timer_end = default_timer()
                time_taken = timer_end - timer_start
                time_taken = '{0:5f} s'.format(round(time_taken, 5))
                defined_logger.debug('Time taken to execute {} function is [{}]'.format(func.__qualname__, time_taken))
                defined_logger.debug('The function {} have returned {}'.format(func.__qualname__, ret))
                return ret
            except Exception as err:
                # log the exception
                msg = "There was '{}' exception in  {}\n{}".format(type(err).__name__, func.__qualname__, err)
                defined_logger.exception(msg)
                # re-raise the exception
                raise err

        return log_wrapper
    return decorator

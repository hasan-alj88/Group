import functools
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

default_timer = time.time


def create_logger(name=None):
    name = name or __name__
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)

    # Create logs directory using pathlib
    log_dir = Path(__file__).parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / 'GroupPythonProject.log'

    # Stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)

    # log file handler
    file_handler = RotatingFileHandler(
        filename=str(log_file),  # Convert Path to string for RotatingFileHandler
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)

    # create formatter
    log_format = logging.Formatter('%(levelname)s||%(funcName)s||%(asctime)s||%(message)s')

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
                defined_logger.debug(f'{func.__qualname__}({args}) function will start.')
                timer_start = default_timer()
                ret = func(*args, **kwargs)
                timer_end = default_timer()
                time_taken = timer_end - timer_start
                time_taken = '{0:5f} s'.format(round(time_taken, 5))
                defined_logger.debug(f'Time taken to execute {func.__qualname__} function is [{time_taken}]')
                defined_logger.debug(f'The function {func.__qualname__} have returned {ret}')
                return ret
            except Exception as err:
                # log the exception
                msg = f"There was '{type(err).__name__}' exception in {func.__qualname__}\n{err}"
                defined_logger.exception(msg)
                # re-raise the exception
                raise err

        return log_wrapper

    return decorator
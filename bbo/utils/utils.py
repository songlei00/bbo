import time
import logging

logger = logging.getLogger(__name__)


def timer_wrapper(func):
    def wrapper(*args, **kwargs):
        st = time.monotonic()
        ret = func(*args, **kwargs)
        et = time.monotonic()
        logger.info('func: ' + func.__name__ + ', time: {} sec'.format(et - st))
        return ret
    return wrapper
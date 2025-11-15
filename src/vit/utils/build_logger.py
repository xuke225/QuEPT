import logging
import os
import sys


def get_logger(dir_path):
    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_path = os.path.join(dir_path + 'logger.log')
    logger_name = "QuEPT-VIT" 
    logger = logging.getLogger(logger_name)
    
    log_format = '[%(name)s]|[%(levelname)s]|%(asctime)s : %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M %p')
    
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger
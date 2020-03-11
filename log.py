#!/usr/bin/env python3
import logging


def get_logger(prefix="ML_exp"):
    logging.basicConfig(
        format="%(asctime)s %(levelname)s {thd:%(thread)d} [%(filename)s => %(lineno)d] : %(message)s"
    )
    logger = logging.getLogger(prefix)
    logger.setLevel(logging.INFO)
    return logger

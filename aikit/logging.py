# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 10:25:01 2018

@author: Lionel Massoulard
"""

import time
import datetime

import logging
from logging import FileHandler


def _set_logging_to_console(aikit_only=True):
    """ set aikit log to display in the console """
    if aikit_only:
        logger = logging.getLogger("aikit")
    else:
        logger = logging.getLogger()

    # DEBUG level so that everything is written
    logger.setLevel(logging.DEBUG)

    # creation of a second handler that redirect everything on the console

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)

    if not any(
        [isinstance(s, logging.StreamHandler) for s in logger.handlers]
    ):  # To add the handler only one time (in case of reload)
        logger.addHandler(stream_handler)


# _set_logging_to_console(aikit_only = True)

logger = logging.getLogger(__name__)


def log_something_to_test(something="something"):
    logger.info(something)


def register_log_files(log_files, aikit_only=True):
    """ helper to add a new file to log in """

    if isinstance(log_files, str):
        log_files = [log_files]

    if aikit_only:
        logger = logging.getLogger("aikit")
    else:
        logger = logging.getLogger()

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s :: %(levelname)s :: %(message)s")

    for log_file in log_files:
        file_handler = FileHandler(log_file, mode="w")
        # on lui met le niveau sur DEBUG, on lui dit qu'il doit utiliser le formateur
        # créé précédement et on ajoute ce handler au logger
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return None


def display_progress(i=None, N=None, p=1, estimate_remaining_time=False):
    """ in a loop : for i in xrange(10):display_progress(i,10,1) """
    if N is None:

        if i is not None:
            p = i

        if not hasattr(display_progress, "__last_loop"):
            setattr(display_progress, "__last_loop", None)

        if not hasattr(display_progress, "__counter"):
            setattr(display_progress, "__counter", 0)

        tnow = time.time()
        counter = display_progress.__counter

        if display_progress.__last_loop is None:
            display_progress.__last_loop = tnow

        last_loop = display_progress.__last_loop
        if last_loop is not None and counter % p == 0:
            logger.info(" -- since last loop : %2.2f secs --" % (tnow - last_loop))
            display_progress.__last_loop = tnow

        display_progress.__counter += 1

    else:
        if i == 0:
            if not hasattr(display_progress, "__start_of_loop"):
                setattr(display_progress, "__start_of_loop", list())
            display_progress.__start_of_loop.append(time.time())

        if i % p == 0 or i == N - 1 or i == 0:
            if estimate_remaining_time:
                elapsed_time_since_start = time.time() - display_progress.__start_of_loop[-1]
                remaining_time = (elapsed_time_since_start / (i + 1)) * (N - (i + 1))
                estimated_end_time = datetime.datetime.now() + datetime.timedelta(0, remaining_time)
                msg = "Elapsed time : %2.2f secs -- Remaining Time : %2.2f -- End Time : %s" % (
                    elapsed_time_since_start,
                    remaining_time,
                    estimated_end_time.strftime("%H:%M"),
                )
            else:
                msg = ""

            # sys.stdout.write("\r                                             ")
            # sys.stdout.write("\r -- %d/%d -- %s" % (i,N-1,msg))
            logger.info(" -- %d/%d -- %s" % (i, N - 1, msg))

        if i == N - 1 and len(display_progress.__start_of_loop):
            start_of_loop = display_progress.__start_of_loop.pop()
            tnow = time.time()
            logger.info(" -- Elapsed time : %2.2f secs --" % (tnow - start_of_loop))

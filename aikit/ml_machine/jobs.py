# -*- coding: utf-8 -*-
"""
Created on Fri May 11 17:40:13 2018

@author: Lionel Massoulard
"""

import logging

logger = logging.getLogger(__name__)

from aikit.logging import register_log_files
import os.path

import gc
import time

try:
    import msvcrt
except ModuleNotFoundError:
    # dont work on linux apparently
    # https://stackoverflow.com/questions/24072790/detect-key-press-in-python
    msvcrt = None

import numpy as np
from sklearn.utils import check_random_state

from aikit.ml_machine.data_persister import SavingType
from aikit.tools.helper_functions import system_and_caller_information, md5_hash


class AbstractJobRunner(object):
    """ Class to handle jobs """

    worker_path = None

    def __init__(
        self,
        data_persister,
        gc_collect_freq=1,
        draw_random_for_queue=False,  # Entry queue
        max_queue_waiting_time=-1,
        max_done_queue_waiting_time=None,  # Exit Queue
        input_queue_sleeping_time=1,  # Number of seconds to wait before trying the queue again (when it is full)
        done_queue_sleeping_time=1,  # Number of seconds to wait before trying the queue again (when it is full)
        seed=None,
    ):

        self._all_times = []

        if gc_collect_freq is not None:
            self.gc_collect_freq = gc_collect_freq
        else:
            self.gc_collect_freq = np.inf

        self.data_persister = data_persister

        self.draw_random_for_queue = draw_random_for_queue
        self.max_queue_waiting_time = max_queue_waiting_time
        self.max_done_queue_waiting_time = max_done_queue_waiting_time

        self.input_queue_sleeping_time = input_queue_sleeping_time
        self.done_queue_sleeping_time = done_queue_sleeping_time

        self.seed = seed
        self.random_state = seed

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, new_random_state):
        self._random_state = check_random_state(new_random_state)

    @classmethod
    def get_worker_path(cls):
        if cls.worker_path is None:
            return cls.__name__.lower() + "_workers"
        else:
            return cls.worker_path

    def _prepare_workers(self):

        #################
        ### Worker ID ###
        #################
        self.nb_of_workers = self.data_persister.new_shared_integer(path=self.get_worker_path(), key="nb_of_workers")

        new_nb_of_worker = self.nb_of_workers.inc()

        self.worker_informations = system_and_caller_information()
        self.worker_informations["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.worker_informations["started"] = False
        self.worker_informations["done"] = False
        self.worker_informations["ctrq_pressed"] = False
        self.worker_informations["id"] = "worker_%d" % new_nb_of_worker

        self.worker_informations["class_name"] = self.__class__.__name__

        ### Create random_state ###
        if self.seed is None:
            seed = int(md5_hash(self.worker_informations), 16) % (2 ** 32 - 1)
        else:
            seed = self.seed
        self.worker_informations["seed"] = seed
        self.random_state = check_random_state(seed)  # will create a generator here

        # Peut etre qu'il faut mettre le nomber de l'object
        self.write_worker_informations()

        log_file = os.path.join(
            self.data_persister.base_folder,
            self.get_worker_path(),
            type(self).__name__.lower() + "_worker_%d.log" % new_nb_of_worker,
        )
        register_log_files(log_file, aikit_only=False)

    def write_worker_informations(self):
        self.data_persister.write(
            data=self.worker_informations,
            path=self.get_worker_path(),
            key=self.worker_informations["id"],
            write_type=SavingType.json,
        )

        return self

    #############################################################
    #### Method to implement or overide in inherited classes ####
    #############################################################
    def prepare(self):
        """ implement this method in inherited class to do a custum preparation before running """
        return self

    def do_job(self, job_id, job_param):
        """ implement this method in inherited class to the job """
        raise NotImplementedError("should be implemented in inherited class")

    def get_job_queue(self):
        ### Result should have a 'remove' method ####
        return self.data_persister.new_queue(
            path="job_queue", write_type=SavingType.json, random=self.draw_random_for_queue
        )

    def get_job_done(self):
        ### Result should have an 'add' method ###
        return self.data_persister.new_queue(
            path="job_done", write_type=SavingType.json, max_queue_size=None, random=False
        )

    #############################################################
    #############################################################
    #############################################################

    def must_stop(self):
        """ because of stop file """
        global_stop = self.data_persister.exists(path=self.get_worker_path(), key="stop", write_type=SavingType.txt)
        if global_stop:
            return True, "global_stop"

        worker_stop = self.data_persister.exists(
            path=self.get_worker_path(), key=str(self.worker_informations["id"]) + "_stop", write_type=SavingType.txt
        )
        if worker_stop:
            return True, "worker_stop"

        if self._keyboard_funct() == 17:
            return True, "ctrq_pressed"

        return False, None

    def run(self):
        for job_id, job_param in self.iterate():
            self.do_job(job_id, job_param)

    @staticmethod
    def _keyboard_funct():
        # https://stackoverflow.com/questions/24072790/detect-key-press-in-python
        if msvcrt is not None:
            x = msvcrt.kbhit()
            if x:
                ret = ord(msvcrt.getch())
            else:
                ret = 0
            return ret
        else:
            return 0

    def iterate(self):
        """ main method to iterate using the object """
        self._prepare_workers()
        self.prepare()

        self.job_queue = self.get_job_queue()
        self.job_done = self.get_job_done()

        self.worker_informations["started"] = True
        self.write_worker_informations()
        # Ici : enregistrer son worker

        GARGAGE_COUNT = 0
        while True:

            #############################################
            ### Try to retrieve a job_id in the queue ###
            #############################################
            _start_time_queue = time.time()
            must_stop = False

            do_print = True
            while True:

                job_id = self.job_queue.remove()

                # Ici : on peut peut etre verifier si on a pas deja fait le job
                # ce qui peut arriver, si on a mal synchroniser en entree ? => Ex : on a relancer le controller avec ces models par default ?
                # ou si on a retirer 2 fois un model random,

                if job_id is not None:
                    # I have found something in the queue
                    break

                must_stop, reason = self.must_stop()
                if must_stop:
                    break

                current_time = time.time()
                if (
                    self.max_queue_waiting_time is not None
                    and current_time - _start_time_queue >= self.max_queue_waiting_time
                ):
                    logger.info("queue was empty...")
                    logger.info("stop waiting for queue")
                    break
                else:
                    if do_print:
                        logger.info("queue was empty...")
                        logger.info("wait for queue for %d sec(s)" % self.input_queue_sleeping_time)
                        do_print = False  # to print only time
                    time.sleep(self.input_queue_sleeping_time)

                ###########################################
                # max_queue_waiting_time :                #
                #  * None <=> inf => wait forever         #
                #  * -1           => don't wait at all    #
                #  * x            => wait x seconds       #
                ###########################################

            if job_id is None:
                self.worker_informations["stopped"] = True
                self.worker_informations["stopping_reason"] = "empty queue"
                self.write_worker_informations()

                break

            if must_stop:
                self.worker_informations["stopped"] = True
                self.worker_informations["stopping_reason"] = reason
                self.write_worker_informations()
                logger.info("I must stop because %s" % reason)
                break

            ###########################################
            ### Retrieve the parameters of that job ###
            ###########################################
            job_param = self.data_persister.read(key=job_id, path="job_param", write_type=SavingType.json)

            logger.info("start job_id : %s" % job_id)
            logger.info("")

            try:
                _success = False
                start_time = time.time()

                #################################
                ### Send job_id and job_param ###
                #################################
                yield job_id, job_param

                return_time = time.time()
                _success = True

            finally:

                if not _success:
                    ####################################
                    ### It means there were an error ###
                    ####################################
                    self.worker_informations["stopped"] = True
                    self.worker_informations["stopping_reason"] = "error"
                    self.write_worker_informations()

            ########################
            ### Save time of job ###
            ########################
            self._all_times.append(return_time - start_time)

            ##################################
            ### Do a garbage collector run ###
            ##################################
            GARGAGE_COUNT += 1
            if GARGAGE_COUNT >= self.gc_collect_freq:
                GARGAGE_COUNT = 0
                gc.collect()

            ###############################
            ### Add job to 'done queue' ###
            ###############################
            could_add = False
            _start_time_done_queue = time.time()

            do_print = True
            while True:
                could_add = self.job_done.add(data=job_id)

                if could_add:
                    break

                must_stop, reason = self.must_stop()
                if must_stop:
                    break

                current_time = time.time()
                if (
                    self.max_done_queue_waiting_time is not None
                    and current_time - _start_time_done_queue >= self.max_done_queue_waiting_time
                ):
                    logger.info("done queue was full...")
                    logger.info("stop waiting for done queue")
                    break
                else:
                    if do_print:
                        logger.info("done queue was full...")
                        logger.info("wait for done queue for %d sec(s)" % self.done_queue_sleeping_time)

                        do_print = False  # to print only once
                    time.sleep(self.done_queue_sleeping_time)

                #############################################
                # max_done_queue_waiting_time  :            #
                #  * None <=> inf : wait for ever           #
                #  * -1           : don't wait at all       #
                #  * x            : wait for x seconds      #
                #############################################

            # Ici : regarder si on a un flag 'stop'

            if not must_stop:
                must_stop, reason = self.must_stop()

            if must_stop:
                self.worker_informations["stopped"] = True
                self.worker_informations["stopping_reason"] = reason
                self.write_worker_informations()
                logger.info("I must stop because %s" % reason)
                break

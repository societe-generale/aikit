# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 16:10:17 2018

@author: Lionel Massoulard
"""
import time
import argparse
from enum import Enum


import pandas as pd
from multiprocessing import Process


from aikit.ml_machine.ml_machine import AutoMlConfig, JobConfig, MlJobManager, MlJobRunner, AutoMlResultReader
from aikit.ml_machine.ml_machine_guider import AutoMlModelGuider
from aikit.ml_machine.data_persister import FolderDataPersister, SavingType

from aikit.tools.helper_functions import function_has_named_argument
from aikit.model_definition import sklearn_model_from_param


def start_runner(runner):
    """ helper function to start a running object """
    runner.run()
    return None


class MlMachineLauncher(object):
    """ object to handle the ml machine processes,

    it will hold the configurations,
    and allow the creation of 
    * worker(s)
    * controller
    * creation of results
    * ...

    """

    class Commands(Enum):
        """ enumeration of accepted commands """

        run = "run"
        init = "init"
        worker = "worker"
        controller = "controller"
        stop = "stop"
        result = "result"
        fit = "fit"

    def __init__(self, base_folder, name=None, loader=None, set_configs=None):

        self.base_folder = base_folder
        self.name = name

        self.job_runner = None
        self.job_controller = None

        self.result_reader = None
        self.auto_ml_guider = None
        self.data_persister = None

        self.job_config = None
        self.auto_ml_config = None

        self.loader = loader
        self.set_configs = set_configs

        self.dfX = None
        self.y = None
        self.groups = None

        self._seed = None  # to initialize the attribute
        self._nbworks = None

        self.command = None

    @staticmethod
    def _get_args_parse():
        parser = argparse.ArgumentParser(description="Ml Machine launcher")
        parser.add_argument(
            "command", help="choice among 'run', 'init', 'controller', 'worker', 'stop', 'result', 'fit'", type=str
        )
        parser.add_argument("--nbworkers", "-n", help="number of workers to start", type=int, default=1)
        parser.add_argument("--seed", help="force seed of worker(s) or controllers", type=int, default=None)
        parser.add_argument("--job_ids", help="the job_id(s) of the model to fit")

        return parser

    def _process_command_arguments(self, parser=None):
        """ process the command arguments to save in the manager the command to launch and its parameters """

        if parser is None:
            parser = self._get_args_parse()

        if not hasattr(parser, "parse_args"):
            raise TypeError("'parser' should have a 'parse_args' method")

        args = parser.parse_args()

        allowed_commands = [c.value for c in self.Commands]
        if args.command is None or args.command.lower() not in allowed_commands:
            raise ValueError("Unknown command please choose among %s" % ", ".join(allowed_commands))

        self.command = self.Commands(args.command.lower())
        self._nbworkers = args.nbworkers
        self._seed = args.seed
        self._job_ids = args.job_ids
        # self._noinit = args.noinit

        return self

    def execute_processed_command_argument(self, parser=None):

        self._process_command_arguments(parser)

        if self.command == self.Commands.run:
            self.run_command()

        elif self.command == self.Commands.init:
            self.init_command()

        elif self.command == self.Commands.stop:
            self.stop_command()

        elif self.command == self.Commands.worker:
            self.worker_command()

        elif self.command == self.Commands.controller:
            self.controller_command()

        elif self.command == self.Commands.result:
            self.result_command()

        elif self.command == self.Commands.fit:
            self.fit_command(job_ids=[s.strip() for s in self._job_ids.split(",")])
        else:
            raise ValueError("Unknown command %s" % self.command.value)

    def init_command(self):
        """ this command is to initialized a new task WITHOUT any controller and worker.
        This command can be executed using the 'init' keyword in the command argument.
        
        It will :
            * load data base using self.loader, if dfX and y are not present
            * initialize the configurations (if 'initialize_done' is not True)
            * initialize the persister
            * call 'launcher_modif' to apply use change
            * persist everyting so that it can be reloaded later
        
        """
        print("...starting INIT command...")
        print("...initialized")
        self.initialize()

        print("...persist")
        self.persist()

    def run_command(self):
        """ this command is to start a new task WITH controller and worker(s)
        It can be executed using the 'start' keyoword in the command argument.
        
        It will :
            * load data base using self.loader, if dfX and y are not present
            * initialize the configurations
            * initialize the persister
            * persister everything so that it can be reloaded later
            
            * start a controller in a SUB process
            * start 'nbworkers - 1' worker(s) in SUB process(es)
            * start the last worker in main process
        
        """
        print("...start command...")
        self.init_command()

        print("...starting controller in subprocess...")
        self.start_controller_subprocess()

        if self._nbworkers > 1:
            print("...starting %d workers in subprocess..." % (self._nbworkers - 1))
            self.start_worker_subprocess(self._nbworkers - 1)

        if self._nbworkers > 0:
            print("...start worker in main process...")
            self.start_worker()

    def worker_command(self):
        """ this command is to add new worker(s) to an existing task.
        It can be executed using the 'worker' keyword in the command argument.
        
        It will :
            * reload configurations and data
            * start 'nbworkers - 1' worker(s) in SUB process(es)
            * start the last worker in main process
        """
        print("...worker command...")

        print("...reload...")
        self.reload()

        if self._nbworkers > 1:
            print("...starting workers in subprocess...")
            self.start_worker_subprocess(self._nbworkers - 1)

        print("...start worker in main process...")
        self.start_worker()

    def controller_command(self):
        """ this command is to start a controller on an existing task.
        It can be executed using the 'controller' keyword in the command argument.
        
        It will:
            * reload configurations and data
            * start a controller in main process
        
        """

        print("...controller command...")
        self.reload()

        self.start_controller()

    def stop_command(self):
        """ this command is to stop the automl processes.
        It can be executed using the 'stop' keyword in the command argument.
        
        It will:
            * create stop file in controller/worker folder to stop them
        
        """

        self.data_persister = FolderDataPersister(base_folder=self.base_folder)
        self.data_persister.write("", key="stop", path="mljobmanager_workers", write_type=SavingType.txt)
        self.data_persister.write("", key="stop", path="mljobrunner_workers", write_type=SavingType.txt)

    def result_command(self):
        """ this command is to launch aggregat of result.
        It can be executed using the 'result' keyword in the command argument.
        
        It will:
            * load result, params and error
            * merge them
            * save everything into two excel files
            
        """
        self.data_persister = FolderDataPersister(base_folder=self.base_folder)
        self.result_reader = AutoMlResultReader(self.data_persister)

        df_results = self.result_reader.load_all_results()
        df_additional_results = self.result_reader.load_additional_results()
        df_params = self.result_reader.load_all_params()
        df_errors = self.result_reader.load_all_errors()

        df_params_other = self.result_reader.load_all_other_params()

        df_merged_result = pd.merge(df_params, df_results, how="inner", on="job_id")
        df_merged_result = pd.merge(df_merged_result, df_params_other, how="inner", on="job_id")
        if df_additional_results.shape[0] > 0:
            df_merged_result = pd.merge(df_merged_result, df_additional_results, how="inner", on="job_id")

        df_merged_error = pd.merge(df_params, df_errors, how="inner", on="job_id")

        #        df_merged_result2 = pd.merge( df_params_other, df_results, how = "inner",on = "job_id")
        #        df_merged_result2 = df_merged_result2.sort_values(by="job_creation_time")

        try:
            df_merged_result.to_excel(self.base_folder + "/result.xlsx", index=False)
            print("file %s saved" % self.base_folder + "/result.xlsx")
        except OSError:
            print("I couldn't save excel file")

        try:
            df_merged_error.to_excel(self.base_folder + "/result_error.xlsx", index=False)
            print("file %s saved" % self.base_folder + "/result_error.xlsx")
        except OSError:
            print("I couldn't save excel file")

        return df_merged_result, df_merged_error

    def fit_command(self, job_ids):
        """ this command is to launch the final fit one (or more) model(s)
        It can be executed using the 'fit' command keyword followed by '--job_ids ***'
        
        It will:
            * reload the data
            * fit a model on all the data
            * save the pickled object

        """
        all_models = []
        for job_id in job_ids:
            print("fitting of job_id '%s'" % job_id)
            self.reload()

            job_param = self.data_persister.read(job_id, path="job_param", write_type=SavingType.json)
            model = sklearn_model_from_param(job_param["model_json"])
            print("start fitting...")

            if function_has_named_argument(model.fit, "groups") and self.groups is not None:
                model.fit(self.dfX, self.y, groups=self.groups)
            else:
                model.fit(self.dfX, self.y)

            print("...model fitted!")

            self.data_persister.write(model, job_id, path="saved_models", write_type=SavingType.pickle)
            self.data_persister.write(job_param["model_json"], job_id, path="saved_models", write_type=SavingType.json)

            print("model persisted")

            all_models.append(model)

        return all_models

    # In[]
    def reload(self):
        """ method to reload dfX, y, auto_ml_config and job_config """
        self.data_persister = FolderDataPersister(base_folder=self.base_folder)

        self.job_config = self.data_persister.read(key="job_config", write_type=SavingType.pickle)

        self.auto_ml_config = self.data_persister.read(key="auto_ml_config", write_type=SavingType.pickle)
        self.dfX = self.data_persister.read(key="dfX", write_type=SavingType.pickle)
        self.y = self.data_persister.read(key="y", write_type=SavingType.pickle)
        self.groups = self.data_persister.read(key="groups", write_type=SavingType.pickle)

    def initialize(self):
        """ method to initialize auto_ml_config and job_config """

        ##################################
        ### ** load data and target ** ###
        ##################################
        if self.dfX is None or self.y is None:
            temp = self.loader()
            if len(temp) == 2:
                self.dfX, self.y = temp
                self.groups = None
            else:
                self.dfX, self.y, self.groups = temp

        ###########################################
        ### ** create database configuration ** ###
        ###########################################
        if self.auto_ml_config is None:
            self.auto_ml_config = AutoMlConfig(dfX=self.dfX, y=self.y, groups=self.groups, name=self.name)
            self.auto_ml_config.guess_everything()

        #####################################
        ### ** create job configuation ** ###
        #####################################
        if self.job_config is None:
            self.job_config = JobConfig()
            self.job_config.guess_cv(auto_ml_config=self.auto_ml_config, n_splits=10)
            self.job_config.guess_scoring(auto_ml_config=self.auto_ml_config)

        ###################################
        ### ** create data persister ** ###
        ###################################
        if self.data_persister is None:
            self.data_persister = FolderDataPersister(base_folder=self.base_folder)

        #################################
        ### ** apply custom config ** ###
        #################################
        if self.set_configs is not None:
            self.set_configs(self)

    def persist(self):
        """ method to persist 'auto_ml_config', 'job_config', 'dfX' and 'y' """
        ##########################################
        ### ** Persist everything if needed ** ###
        ##########################################

        self.auto_ml_config.dfX = None
        self.auto_ml_config.y = None
        self.auto_ml_config.groups = None

        self.data_persister.write(data=self.job_config, key="job_config", write_type=SavingType.pickle)
        self.data_persister.write(data=self.auto_ml_config, key="auto_ml_config", write_type=SavingType.pickle)
        self.data_persister.write(data=self.dfX, key="dfX", write_type=SavingType.pickle)
        self.data_persister.write(data=self.y, key="y", write_type=SavingType.pickle)
        self.data_persister.write(data=self.groups, key="groups", write_type=SavingType.pickle)

    ###############
    ## Controler ##
    ###############

    def _create_controller(self):
        """ create a controller object, if it doesn't exist, but doesn't start it """
        ###############################
        ### ** Create controller ** ###
        ###############################
        if self.job_controller is None:

            self.result_reader = AutoMlResultReader(self.data_persister)
            self.auto_ml_guider = AutoMlModelGuider(
                result_reader=self.result_reader,
                job_config=self.job_config,
                metric_transformation="default",
                avg_metric=True,
            )

            self.job_controller = MlJobManager(
                auto_ml_config=self.auto_ml_config,
                job_config=self.job_config,
                auto_ml_guider=self.auto_ml_guider,
                data_persister=self.data_persister,
                seed=self._seed,
            )

        return self

    def start_controller(self):
        """ create a controller and start it in MAIN process """
        self._create_controller()
        self.job_controller.run()

    def start_controller_subprocess(self):
        """ create a controller and start it in a SUB process """
        self._create_controller()
        self.controller_process = Process(target=start_runner, args=(self.job_controller,))
        self.controller_process.start()

        time.sleep(1)  # to prevent lock issue if everything is created at the same time...

    ############
    ## Worker ##
    ############

    def _create_worker(self):
        """ create a worker, if it doesn't exist, but doesn't start it """
        if self.job_runner is None:

            ##############################
            ### ** Create worker(s) ** ###
            ##############################

            self.job_runner = MlJobRunner(
                dfX=self.dfX,
                y=self.y,
                groups=self.groups,
                auto_ml_config=self.auto_ml_config,
                job_config=self.job_config,
                data_persister=self.data_persister,
                seed=self._seed,
            )

        return self

    def start_worker(self):
        """ create a worker and start it in MAIN process """
        self._create_worker()

        self.job_runner.run()

    def start_worker_subprocess(self, n=1):
        """ create a worker and start 'n' copy of it in SUB processes """
        self._create_worker()

        self.all_job_runner_process = []
        for other_worker in range(n):
            self.all_job_runner_process.append(Process(target=start_runner, args=(self.job_runner,)))

        for job_runner_process in self.all_job_runner_process:

            job_runner_process.start()
            time.sleep(1)  # small pause to prevent lock mistakes

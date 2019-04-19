.. _ml_machine_launcher:

Ml Machine Launcher
-------------------

To help to launch the ml machine jobs you can use the :class:`MlMachineLauncher` object.

it :

 * contains the configurations of the auto-ml
 * has methods to launch a controller, a worker, ... (See after)
 * has a method to process command line arguments to quickly create a script that can be used to drive the ml-process (See after)


Example::

    from aikit.datasets import load_dataset, DatasetEnum
    from aikit.ml_machine import MlMachineLauncher

    def loader():
        """ modify this function to load the data

        Returns
        -------
        dfX, y

        Or
        dfX, y, groups

        """
        dfX, y, *_ = load_dataset(DatasetEnum.titanic)
        return dfX, y

    def set_configs(launcher):
        """ modify that function to change launcher configuration """

        launcher.job_config.score_base_line = 0.75
        launcher.job_config.allow_approx_cv = True

        return launcher

    if __name__ == "__main__":
        launcher = MlMachineLauncher(base_folder = "C:/automl/titanic", 
                                     name = "titanic",
                                     loader = loader,
                                     set_configs = set_configs)

        launcher.execute_processed_command_argument()

(in what follows we will assume that this is the content of the *"automl_launcher.py"* file)

Here is what is going on:

1. first import the launcher
2. define a loader function : it is better to define a loading function that can be called **if needed** instead of just loading the data (because you don't always need the data)
3. create a launcher, with a base folder and the loading function
4. (Optional) : you can change a few things in the configurations. Here we set the base line to 75% and tell the auto-ml that it can do approximate cross-validation.
To do that pass a function that change the configuration.
5. Process the command argument to actually start a command

Remarks:

 - if no change in the default configurations are needed you can use set_configs = None
 - the loading function can also return 3 things : dfX, y and groups (if a group cross-validation is needed)
 - don't forget the 'if __name__ == "__main__"' part, since the code uses subprocess it is really needed




Having created a script like that you can now use the script to drive the auto-ml process : 
 * to start a controller and *n* workers
 * to aggregate the result
 * to separately start a controller
 * to separately start worker(s)
 * ...
 
what you need to specify ?
**************************
For the automl to work you need to specify a few things:
 * a loader function
 
This function will load your data, it should return a DataFrame with features (dfX), the target (y), and optionnaly the groups (if you want to use a GroupedCV)
It will be called only once during the initialisation phase. So if you're loading data you don't need to save it a shared folder accessible by all the worker.
(After it is called, the auto-ml will persist everything needed)

 * a base folder : the folder on which the automl will work.
This folder should be accessible by all the workers and the controller.
It will be used to save result, save the queue of jobs, the logs, ...
 
* set_configs function : a function to modify the settings of the automl
You can modify the cv, the base line, the scoring, ...

run command
***********
This is the main command, it will start everything that is needed.
To start the whole process, you should use the 'run' command, in a command windows you can run::

    python automl_launcher.py run
    
This is the main command, it will
    1. load the data using the loader
    2. initialize everything
    3. modify configuration
    4. save everything needed to disk
    5. start one controller in a subprocess
    6. start one worker
    
You can also start more than one worker, to do that, the "-n" command should be used::

    python automl_launcher.py run -n 4

This will create a total of 4 workers (and also 1 controller), so at the end you'll have 5 python processes running


manual start
************

You can also use this script to start everything manually. That way you can 
 * do the initialization manually
 * have one console for the controller
 * have separate consoles for workers

To do that you need the same steps as before.

init command
************

If you only want to initialize everything, you can run the 'init' command::
    
    python automl_launcher.py init

This won't start anything (no worker, no controller), but will load the data, prepare the configuration and apply the change and persist everything to disk.
    
manual init
***********
alternatively you can do that manually in a notebook or your favorite IDE. That way you can actually see what the default configuration, prepare the data, etc.

Here is the code to do that::

     launcher.MlMachineLauncher(base_folder="C:/automl/titanic", loader=loader)
     launcher.initialize()
     launcher.job_config.base_line = 0.75
     launcher.auto_ml_config.columns_informations["Pclass"]["TypeOfVariable"] = "TEXT"
     
     # ... here you can take a look at job_config and auto_ml_config
     # ... any other change
     
     launcher.persist()
     
controller command
******************

If you only want to start a controller, you should use the 'controller' command::

    python automl_launcher.py controller
    
This will start one controller (in the main process)

worker command
**************

If you only want to start worker(s) you should use the 'worker' command::

    python automl_launcher.py worker -n 2
    
This will start 2 workers (one in main process and one in a subprocess). For it to do anything a controller needs to be started elsewhere.
This command is useful to add new workers to an existing task, or to add new worker on another computer (assuming the controller is running elsewhere).


result command
**************

If you want to launch the aggregation of result, you can use the 'result' command::

    python automl_launcher.py result
    
This will trigger the results aggregations and generate the excel result file

stop command
*************

If you want to stop every process, you can use the 'stop' command::
    
    python automl_launcher.py stop
    
It will create the stop file that will trigger the exit of all process listening to that folder


Summary
*******

To start a new experiment, first create the script with the example above then use run command.

If you want to split everything you can use

    1. launcher.initialize()
    2. apply modifications
    3. launcher.persist()
    4. controller command
    5. worker command
    
Whenever you want an aggregation of results : result command


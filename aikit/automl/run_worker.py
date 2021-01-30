from aikit.automl.utils import deactivate_warnings, configure_console_logging
from aikit.automl.launcher import AutoMlLauncher


deactivate_warnings()
configure_console_logging()

path = r'C:\data\automl'
launcher = AutoMlLauncher(path)
launcher.start_worker()

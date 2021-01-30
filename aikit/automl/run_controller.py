import os
from aikit.datasets import load_dataset, DatasetEnum
from aikit.tools.helper_functions import save_pkl
from aikit.automl.utils import deactivate_warnings, configure_console_logging
from aikit.automl.launcher import AutoMlLauncher


deactivate_warnings()
configure_console_logging()

path = r'C:\data\automl'
X, y, *_ = load_dataset(DatasetEnum.titanic)
save_pkl((X, y), os.path.join(path, 'data.pkl'))

launcher = AutoMlLauncher(path)
launcher.start_controller()

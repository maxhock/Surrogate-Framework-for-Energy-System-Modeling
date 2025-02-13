from surrogate_helper.data_handling import *
from surrogate_helper.hypertuning import *
from surrogate_helper.model_handling import *
from surrogate_helper.plotting import *
from surrogate_helper.denormalized_metrics import *

def printConfig(identifier, training, tuning, transfer, verbosity, backend, data_settings, architecture):
    print("##########################")
    print(f"Identifier: {identifier}")
    print(f"Training: {training}")
    print(f"Tuning: {tuning}")
    print(f"Transfer: {transfer}")
    print(f"Verbosity: {verbosity}")
    print(f"Backend: {backend}")
    print(f"Timeseries Window: {data_settings['timeseries_window']}")
    print(f"Timeseries Stride: {data_settings['timeseries_stride']}")
    print(f"Augmentation: {data_settings['augment']}")
    print(f"Architecture: {architecture}")
    print("##########################")

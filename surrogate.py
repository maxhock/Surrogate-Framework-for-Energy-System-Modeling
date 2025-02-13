"""
Batch Version of the surrogate.ipynb notebook
"""

import os
import argparse
from pathlib import Path


## Settings for Dataset
data_settings = {
    "timeseries_window":  24,
    "timeseries_stride": 1,
    "testing_size": 1 / 27,  # how much of the total data to be set aside for testing
    "training_size": 0.8,  # how much of the non-testing data should be used for training
    "normalize": True,  # whether to normalize the data after reading it
    "augment": ("ts","vmd",5),  # whether to augment the data with timestamp "ts", seasonal trend decomposition loess "stl" or variational mode decomposition "vmd"
    "n_scenarios": 27, # how many scenarios are in the collected original dataset
}

# Copy settings from data_settings and only overwrite necessary ones
transferdata_settings = data_settings.copy()
transferdata_settings.update({
    "testing_size": 23 / 24, # how much of the total data to be set aside for testing
    "n_scenarios": 24, # how many scenarios are in the collected transfer dataset
})


identifier = "TrainDeltaLoss_LSTM24_TS-VMD5_fixNScenario-DenseNoActivation"

## Settings for Model
model_settings = {
    "architecture": "lstm",
    "data_file": "./energydata/energydata_original.csv",
    "model_file": "model_" + identifier + ".keras",
    "epochs": 200,
    "log_folder": "logs",
    "name": "model_" + identifier,
}

transfer_identifier = identifier

transfermodel_settings = {
    "data_file": "./energydata/energydata_forchheim.csv",
    "model_file": "forchheim_model_" + transfer_identifier + ".keras",
    "epochs": 200,
    "log_folder": "forchheim_logs",
    "name": "forchheim_model_" + transfer_identifier,
}

hp_default = {
    "n_linear_in": 1,  # 3
    "dropout_linear_in": 0.333,  # 0.22743646633159895
    "depth_linear_in": 160,  # 128
    "n_main": 3,  # 3
    "dropout_main": 0.114,  # 0.11547677862094448
    "depth_main": 832,  # 128
    "n_heads_transformer": 3,
    #"d_k_transformer": 64,
    "n_linear_out": 3,  # 3
    "depth_linear_out": 96,  # 32
    "lr": 3e-4,  # 4.661779530579224e-05
    "max_trials": 100,
    "directory": f"./{model_settings['architecture']}/",
    "project": "tuning",
    "overwrite": True,
    "log_path": f"./{model_settings['architecture']}/logs/tuning",
    "architecture": model_settings['architecture'],
}

## Settings for Flow Control
# whether to train or use weights from previous training
DO_TRAINING = False  # @param {type:'boolean'}
DO_HYPERTUNING = False  # @param {type:'boolean'}
DO_TRANSFER = False # @param {type:'boolean'}
VERBOSITY = 2  # 0: Silent, 1: Verbose, 2: Script logging

os.environ["KERAS_BACKEND"] = "jax"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(VERBOSITY)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--training", action="store_true", default=DO_TRAINING, help="Do training, optionally including tuning"
)
parser.add_argument(
    "--tuning", action="store_true", default=DO_HYPERTUNING, help="Do tuning, also reset training progress"
)
parser.add_argument("--transfer", action="store_true", default=DO_TRANSFER, help="Do transfer")
parser.add_argument(
    "--verbosity", type=int, default=VERBOSITY, help="Set verbosity for training and tuning"
)

args = parser.parse_args()

# overwrite default behaviour with commandline arguments if existent
DO_TRAINING = args.training
DO_HYPERTUNING = args.tuning
DO_TRANSFER = args.transfer
VERBOSITY = args.verbosity


import surrogate_helper as sh
import keras

# when using jax, allow dataparallel processing
if os.environ.get("KERAS_BACKEND") == "jax":
    keras.distribution.set_distribution(keras.distribution.DataParallel())

sh.printConfig(
    identifier,
    DO_TRAINING,
    DO_HYPERTUNING,
    DO_TRANSFER,
    VERBOSITY,
    os.environ.get("KERAS_BACKEND"),
    data_settings,
    model_settings["architecture"],
)

energydata = sh.loadData(model_settings["data_file"])
energydata = sh.augmentData(energydata, data_settings["n_scenarios"], data_settings["augment"])
energydata_train, energydata_val, energydata_test, input_shape = sh.processData(energydata, data_settings)

if DO_TRAINING is True:
    if DO_HYPERTUNING is True:
        hypermodel = sh.hyperTune(energydata_train, energydata_val, hp_default, VERBOSITY)
        sh.storeModel(hypermodel, model_settings["architecture"], model_settings["model_file"])
    model = sh.loadModel(model_settings["architecture"], model_settings["model_file"], input_shape, hp_default)
    model = sh.trainModel(
        model,
        model_settings["architecture"],
        model_settings["name"],
        energydata_train,
        energydata_val,
        model_settings["epochs"],
        model_settings["log_folder"],
        VERBOSITY,
    )
    sh.storeModel(model, model_settings["architecture"], model_settings["model_file"])

model = sh.loadModel(model_settings["architecture"], model_settings["model_file"])
sh.evaluateModel(model, model_settings["architecture"], model_settings["name"], energydata_test)
prediction = sh.predictData(model, model_settings["architecture"], model_settings["name"], energydata_test, plot = True)

transferdata = sh.loadData(transfermodel_settings["data_file"])
transferdata = sh.augmentData(transferdata, transferdata_settings["n_scenarios"], transferdata_settings["augment"])
transferdata_train, transferdata_val, transferdata_test, transfer_input_shape = sh.processData(transferdata, transferdata_settings)

if DO_TRANSFER is True:
    transfer_model = sh.loadModel(model_settings["architecture"], model_settings["model_file"])
    transfer_model = sh.transferModel(transfer_model, transfer_input_shape, model_settings["architecture"], transfermodel_settings["name"], transferdata_train, transferdata_val, transfermodel_settings["epochs"], transfermodel_settings["log_folder"], VERBOSITY)
    transfer_model = sh.finetuneModel(transfer_model, transfer_input_shape, model_settings["architecture"], transfermodel_settings["name"], transferdata_train, transferdata_val, transfermodel_settings["epochs"], transfermodel_settings["log_folder"], VERBOSITY)
    sh.storeModel(transfer_model, model_settings["architecture"], transfermodel_settings["model_file"])

if Path(f"./{model_settings['architecture']}/{transfermodel_settings['model_file']}").is_file():
    transfer_model = sh.loadModel(model_settings["architecture"], transfermodel_settings["model_file"])
    sh.evaluateModel(transfer_model, model_settings["architecture"], transfermodel_settings["name"], transferdata_test)
    prediction = sh.predictData(
        transfer_model, model_settings["architecture"], transfermodel_settings["name"], transferdata_test
    )
else: print("No Transferred Model file found, please transfer a model first.")

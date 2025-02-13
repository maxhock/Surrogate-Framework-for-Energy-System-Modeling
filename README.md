# Surrogate-Framework-for-Energy-System-Modeling
Repository for the paper Surrogate Framework for Energy System Modeling.

This is a framework to create neural network surrogates for energy system models created with [urbs](https://github.com/tum-ens/urbs) but itcan be adapted to other ESM as well.

It is an extension and reimplementation of the work of Max Schulze.

## Concept
This framework creates a transformer or lstm model and trains it on a given dataset of input and output timeseries.
It can also apply transfer learning to adjust an existing model to a different dataset/grid model.

Given a working grid model, multiple variationsof this grid model can be created and processed with urbs.
The resulting data is combined with [[hdf5_reader.py]] to create a varied dataset for this grid model.
The dataset is stored as a zipped csv in ./energydata .
Then, either surrogate.py or surrogate.ipynb is used to preprocess the dataset, create the surrogate, hypertune it, train it, evaluate it, and transfer it.

## Working in LRZ AI
Training the surrogate requires significant computation power, therefore the LRZ AI service can be used to run it.
To be able to work in LRZ AI with this please clone this repository into your local working directory.
On the same directory level follow these instructions to create a virtual machine:

### Creating a virtual machine
1. https://doku.lrz.de/lrz-ai-systems-11484278.html
2. create nvidia ngc account
3. login to LRZ AI via SSH
4. apply for compute node: `salloc --partition=lrz-dgx-1-v100x8,lrz-dgx-1-p100x8,lrz-v100x2,lrz-dgx-a100-80x8 --gres=gpu:1 srun --pty bash`
5. download nvidia container image, for example tensorflow: `enroot import -o base_image.sqsh docker://nvcr.io/nvidia/tensorflow:24.02-tf2-py3`
6. unpack container: `enroot create --name keras3 base_image.sqsh`
7. start container:  `enroot start keras3` 
8. `pip install keras keras-tuner tensorflow[and-cuda] matplotlib -U`
9. `pip install --upgrade "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
10. `exit`
11. export enroot container: `enroot export --output keras3_full.sqsh keras3`

### Running a Job
either batch or interactive
in interactive make sure to use full path for custom image
/dss/dsshome1/08/ga84nem2/keras3_full.sqsh

### Problem Solving
When encountering errors during execution try deleting the best_checkpoint.keras file and the training and tuning folders inside your architecture folder.

"""
Functions related to the model and its usage
"""
import keras
from keras import activations, layers
from keras_nlp.layers import SinePositionEncoding, TransformerEncoder
from keras_tuner import HyperParameters
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import surrogate_helper as sh

# @keras.saving.register_keras_serializable(package="MyLoss")
# def physicsLoss(x=0):
@keras.saving.register_keras_serializable(package="MyLoss")
def loss(y_true, y_pred):
    """Custom loss function that includes a MSE and additional terms.

    Args:
        y_true: True targets
        y_pred: Predicted targets
    """
    mse = keras.ops.square(y_true - y_pred)
    mae = keras.ops.absolute(y_true - y_pred)
    # weight is limited to 1 with clip as values are normalized to dev of 1
    weight = keras.ops.clip(keras.ops.absolute(y_true), 0, 1)
    #    weight = 0
    result = keras.ops.mean(
        weight * mse + (1 - weight) * mae, axis=-1
    )  # Note the `axis=-1`
    # print("x:", x)
    return result

    # return loss


###############################################################################
# Model Building
###############################################################################

# def transformerEncoder(inputs, head_size, num_heads, ff_dim, model_dim, dropout=0):
#     # Attention
#     multihead_output = layers.MultiHeadAttention(
#         key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
#     # Addition & Normalization
#     addnorm_output = layers.LayerNormalization()(inputs + multihead_output)

#     # Feed Forward
#     x = layers.Dense(ff_dim)(addnorm_output)
#     x = layers.ReLU()(x)
#     feedforward_output = layers.Dense(inputs.shape[-1])(x)
#     feedforward_output = layers.Dropout(dropout)(feedforward_output)
#     # Addition & Normalization
#     norm_output = addnorm_output + feedforward_output
#     return layers.LayerNormalization()(norm_output)


# # necessary hyperparameter

#             d_k_transformer = hp.Int(
#                 "d_k_transformer",
#                 min_value=32,
#                 max_value=256,
#                 step=32,
#                 default=hp_default["d_k_transformer"],
#             )

# # necessary Variables

#         enc_vocab_size = 20 # Vocabulary size for the encoder
#         input_seq_length = 75  # Maximum length of the input sequence
#         #h = 8  # Number of self-attention heads
#         #d_k = 64  # Dimensionality of the linearly projected queries and keys
#         d_v = 64  # Dimensionality of the linearly projected values
#         #d_ff = 2048  # Dimensionality of the inner fully connected layer
#         d_model = 128  # Dimensionality of the model sub-layers' outputs
#         #n = 6  # Number of layers in the encoder stack

# # Call in functinoal model
    #x = transformerEncoder(x, d_k_transformer, n_heads_transformer, depth_main, d_model, dropout_main)


def buildModel(hp_default, input_shape):
    """Wrapper for buildHyperModel(hp).

    This allows handing over of default hyperpa rameters and a needed shape input.

    Args:
        hp_default: Default values for the hyperparameters.
            This allows creating a model without prior hypertuning.
        input_shape: Shape of data for input layer
    """

    def buildHypermodel(hp):
        """Build transformer model with given hyperparameters.

        The hyperparameters influence the amount of layers,
        depth and dropout for the dense input, transformer and dense output.
        """

        depth_linear_in = hp.Int(
            "depth_linear_in",
            min_value=64,
            max_value=256,
            step=32,
            default=hp_default["depth_linear_in"],
        )

        dropout_linear_in = hp.Float(
            "dropout_linear_in",
            min_value=0.2,
            max_value=0.4,
            default=hp_default["dropout_linear_in"],
        )

        n_linear_in = hp.Int(
            "n_linear_in",
            min_value=1,
            max_value=4,
            default=hp_default["n_linear_in"],
        )
        
        n_main = hp.Int(
            "n_main",
            min_value=2,
            max_value=4,
            default=hp_default["n_main"],
        )

        dropout_main = hp.Float(
            "dropout_main",
            min_value=0.1,
            max_value=0.3,
            default=hp_default["dropout_main"],
        )
        
        if hp_default["architecture"] == "transformer":
            n_heads_transformer = hp.Int(
                "n_heads_transformer",
                min_value=2,
                max_value=16,
                default=hp_default["n_heads_transformer"],
            )
            
            depth_main = hp.Int(
                "depth_main",
                min_value=64,
                max_value=4096,
                step=256,
                default=hp_default["depth_main"],
            )
            
        elif hp_default["architecture"] == "lstm":
            depth_main = hp.Int(
                "depth_main",
                min_value=64,
                max_value=1024,
                step=256,
                default=hp_default["depth_main"],
            )

        n_linear_out = hp.Int(
            "n_linear_out",
            min_value=1,
            max_value=3,
            default=hp_default["n_linear_out"],
        )

        depth_linear_out = hp.Int(
            "depth_linear_out",
            min_value=32,
            max_value=128,
            step=32,
            default=hp_default["depth_linear_out"],
        )
                
        inputs = keras.Input(shape=input_shape)
        x = inputs

        for _ in range(n_linear_in):
            x = layers.Dense(depth_linear_in, activation=None)(x)
            x = layers.Dropout(dropout_linear_in)(x)

        if hp_default["architecture"] == "transformer":
            # Sine and Cosine encoding for positions, no embedding needed, since this is provided by the dense layers beforehand.
            # x = SinePositionEncoding()(x) + x
            encoding = SinePositionEncoding()(x)
            x = layers.Add()([encoding, x])
            
            for _ in range(n_main):
                x = TransformerEncoder(depth_main, n_heads_transformer, dropout_main)(x)
            x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
            
        elif hp_default["architecture"] == "lstm":
            for _ in range(n_main):
                x = layers.LSTM(depth_main, dropout=dropout_main, return_sequences=True)(x)
            x = layers.LSTM(depth_main, dropout=dropout_main, return_sequences=False)(x)

        for _ in range(n_linear_out):
            x = layers.Dense(depth_linear_out, activation=None)(x)
        outputs = layers.Dense(1)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            #loss="mse",
            loss=loss,
            optimizer=keras.optimizers.Adam(
                learning_rate=hp.Float(
                    "lr",
                    min_value=1e-5,
                    max_value=1e-2,
                    # sampling="log",
                    default=hp_default["lr"],
                ),
            ),
            metrics=[
                keras.metrics.MeanSquaredError(name="mse"),
                keras.metrics.MeanAbsoluteError(name="mean_deviation"),
                keras.metrics.MeanAbsolutePercentageError(name="mape"),
                sh.AggregatedDemand(),
                sh.PeakDemand(),
                sh.PeakFeedIn(),
            ],
        )
        return model

    return buildHypermodel


###############################################################################
# Model Handling
###############################################################################


def storeModel(model, architecture, model_file):
    """Store the model to a file

    Args:
        model: Model to store
        model_file: Path string on where to store the model
    """

    model.save(f"./{architecture}/{model_file}")
    

def loadModel(architecture, model_file, input_shape=None, hp_default=None):
    """Load a model from File

    Args:
        architecture: label of the architecture e.g. transformer, lstm
            Used to identify the storage folder
        model_file: Path string form where to load the model
        input_shape: Shape of the input data to inform the first layer of the model
        hp_default: default hyperparameters if model
            needs to be created from scratch
    """
    try:
        model = keras.saving.load_model(f"./{architecture}/{model_file}")
    except ValueError:
        print("Model file not found. Creating model from default hyperparameters.")
        if hp_default == None or input_shape == None:
            print("No default hyperparameters or input shape given. Cannot create model")
            raise ValueError("No default hyperparameters or input shape given. Cannot create model")
        model = buildModel(hp_default, input_shape)(HyperParameters())
    return model


###############################################################################
# Model Training
###############################################################################


def trainModel(model, architecture, label, data_train, data_val, epochs: int, log_folder, verbosity):
    """Train given model on data for so many epochs.

    All logs and data produced are stored inside './transformer/' or a subfolder.
    Once finished a plot of the loss reduction is printed if possible.

    Args:
        model: Model object
        data_train: Dataset used for training
        data_val: Dataset used for validation
        epochs: Max amount of epochs to train the model for.
            EarlyStopping can still result in earlier conclusion.
        log_folder: folder string for storing tensorboard logs
        verbosity: amount of logging output

    Returns:
        Trained model.
    """

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.9, patience=2, min_lr=1e-5
    )

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=f"./{architecture}/best_checkpoint.keras",
        monitor="loss",
        mode="min",
        save_best_only=True,
        verbose=True,
    )

    earlystopping = keras.callbacks.EarlyStopping(
        monitor="loss",
        patience=20,
        restore_best_weights=True,
    )

    backupandrestore = keras.callbacks.BackupAndRestore(
        backup_dir=f"./{architecture}/training",
    )

    tensorboard = keras.callbacks.TensorBoard(f"{architecture}/{log_folder}")

    #    for _ in range(2):
    #        try:
    history = model.fit(
        data_train,
        validation_data=data_val,
        callbacks=[
            reduce_lr,
            checkpoint,
#            earlystopping,
            backupandrestore,
            tensorboard,
        ],
        # use_multiprocessing=True,
        shuffle=True,
        epochs=epochs,
        verbose=verbosity,
    )
    #        except ValueError:
    #            print("Deleting stored weights of BackupAndRestore, then trying again.")
    #            Path("./lstm/training/latest.weights.h5").unlink()
    #            continue
    #        break
    
    
    for x in ("loss","mse"):
        plt.plot(history.history[x], label="Training")
        plt.plot(history.history[f"val_{x}"], label="Validation")
        plt.ylabel(x)
        plt.xlabel("epoch")
        plt.legend()
        Path(f"./{architecture}/{log_folder}/diagrams").mkdir(parents=True, exist_ok=True)
        plt.savefig(f"./{architecture}/{log_folder}/diagrams/{label}-{x}.pdf", bbox_inches='tight')
        plt.show()
        plt.close()
        
    return model



###############################################################################
# Model Evaluation
###############################################################################
def evaluateModel(model, folder: str, title: str, data_test):
    """Evaluate a given Model on the provided dataset and print metrics.

    Args:
        model: Model object to be evaluated
        folder: Folder to store the plots in
        title: name for plots
        data_test: Dataset the model is evaluated on
    """

    metrics = model.evaluate(data_test)

    metrics = dict(
        zip(
            (
                "loss",
                "mse",
                "mean_deviation",
                "mape",
                "aggregated_demand [%]",
                "peak_demand [%]",
                "peak_feed_in [%]",
            ),
            metrics,
        )
    )

    print("##########################")
    for key, value in metrics.items():
        print(f"{key}: {value:0.3f}")
    print("##########################")
        
    #sh.plotActivations(model, folder, title, data_test)


def predictData(model, folder: str, title: str, data_test, plot = False):
    """Predict target data with a model and print the results.

    This function uses a model to predict a target as defined in the dataset.
    It then plots all of the true targets included in the dataset
    as well as the first 24*7 entries for a detailed view.

    Args:
        model: Model object to use for prediction
        folder: Folder to store the plots in
        title: name for plots
        data_test: Dataset to be used for prediction

    Returns:
        Prediction
    """

    prediction = model.predict(data_test)
    
    print(f"Prediction Shape: {prediction.shape}")
    
    #test = data_test.unbatch()
    # features = list(test.map(lambda x, y: x))
    #targets = list(test.map(lambda x, y: y))
    #target_values = []
    #for target in targets:
    #    target_values.append(target.numpy())
        
    target_values = []
    for _, target in data_test.unbatch().as_numpy_iterator():
        target_values.append(target)
    target_values = np.array(target_values)
    
    print(f"Target Shape: {target_values.shape}")

    #import pdb; pdb.set_trace()
    if plot == True:
        sh.plotData(target_values, prediction, folder, title, ("full","week","day"))
        sh.plotScatter(target_values, prediction, folder, title)
    
    return prediction
            

def transferModel(model, input_shape, architecture, label, data_train, data_val, epochs, log_folder, verbosity, lr=1e-2):
    """Freezes all but the last 2 layers of a model and trains it on the given data.

    Args:
        model: Model object
        data_train: Transfer training dataset
        data_val: Transfer validation dataset
        epochs: Epochs to train the transferred model for
        log_folder: Path for where to store the logs generated during training
        verbosity: Amount of logging output

    Returns:
        Transferred model
    """

    model.trainable = False
    
    #for layer in model.layers[-2:]:
    #    layer.trainable = True
    #transferred_model = model
    
    output_dense_config = model.layers[-2].get_config()
    
    x = layers.Dense(output_dense_config["units"], activation=output_dense_config["activation"], name=output_dense_config["name"]+"_transfer")(model.layers[-3].output)
    
    outputs = layers.Dense(1, name="dense_output_transfer")(x)
    
    transferred_model = keras.Model(inputs=model.inputs, outputs=outputs)
        
    transferred_model.compile(
        loss=loss,
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        metrics=[
            keras.metrics.MeanSquaredError(name="mse"),
            keras.metrics.MeanAbsoluteError(name="mean_deviation"),
            keras.metrics.MeanAbsolutePercentageError(name="mape"),
            sh.AggregatedDemand(),
            sh.PeakDemand(),
            sh.PeakFeedIn(),
        ],
    )
    #transferred_model.summary()
    transferred_model = trainModel(transferred_model, architecture, label, data_train, data_val, epochs, log_folder, verbosity)
    
    return transferred_model

def finetuneModel(model, input_shape, architecture, label, data_train, data_val, epochs, log_folder, verbosity):
    model.trainable = True
    model.compile(
        loss=loss,
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=[
            keras.metrics.MeanSquaredError(name="mse"),
            keras.metrics.MeanAbsoluteError(name="mean_deviation"),
            keras.metrics.MeanAbsolutePercentageError(name="mape"),
            sh.AggregatedDemand(),
            sh.PeakDemand(),
            sh.PeakFeedIn(),
            ],
    )
    
    finetuned_model = trainModel(
        model, architecture, label, data_train, data_val, epochs, log_folder, verbosity
    )
    return finetuned_model

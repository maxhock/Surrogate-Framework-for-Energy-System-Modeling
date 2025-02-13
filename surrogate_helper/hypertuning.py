"""
Functions related to hypertuning
"""
import keras
import keras_tuner
import surrogate_helper as sh

@keras.saving.register_keras_serializable(package="MyMetrics")
class PeakDemand(keras.metrics.Metric):
    """Calculate the difference between max of `y_pred`and `y_true` over max `y_true`.

    This indicates whether the prediction lags compared to the truth during demand.
    """

    def __init__(self, name="peak_demand", **kwargs):
        super().__init__(name=name, **kwargs)
        self.max_pred = self.add_weight(name="max_pred", initializer="zeros")
        self.max_true = self.add_weight(name="max_true", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.max_pred.assign(keras.ops.maximum(keras.ops.max(y_pred), self.max_pred))
        self.max_true.assign(keras.ops.maximum(keras.ops.max(y_true), self.max_true))

    def result(self):
        return 100 * keras.ops.abs(1 - (self.max_pred / self.max_true))

    def reset_states(self):
        self.max_pred.assign(0)
        self.max_true.assign(0)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}


@keras.saving.register_keras_serializable(package="MyMetrics")
class PeakFeedIn(keras.metrics.Metric):
    """Calculate the difference between min of `y_pred`and `y_true` over min `y_true`.

    This indicates whether the prediction lags compared to the truth during feedin.
    """

    def __init__(self, name="peak_feed_in", **kwargs):
        super().__init__(name=name, **kwargs)
        self.min_pred = self.add_weight(name="min_pred", initializer="zeros")
        self.min_true = self.add_weight(name="min_true", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.min_pred.assign(keras.ops.minimum(keras.ops.min(y_pred), self.min_pred))
        self.min_true.assign(keras.ops.minimum(keras.ops.min(y_true), self.min_true))

    def result(self):
        return 100 * keras.ops.abs(1 - (self.min_pred / self.min_true))

    def reset_states(self):
        self.min_pred.assign(0)
        self.min_true.assign(0)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}


@keras.saving.register_keras_serializable(package="MyMetrics")
class AggregatedDemand(keras.metrics.Metric):
    """Calculate the difference between sums of `y_pred`and `y_true` over sum `y_true`."""

    def __init__(self, name="aggregated_demand", **kwargs):
        super().__init__(name=name, **kwargs)
        self.sum_pred = self.add_weight(name="sum_pred", initializer="zeros")
        self.sum_true = self.add_weight(name="sum_true", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.sum_pred.assign_add(keras.ops.sum(y_pred))
        self.sum_true.assign_add(keras.ops.sum(y_true))

    def result(self):
        return 100 * keras.ops.abs(1 - (self.sum_pred / self.sum_true))

    def reset_states(self):
        self.sum_pred.assign(0)
        self.sum_true.assign(0)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}


def hyperTune(data_train, data_val, hp_default, verbosity):
    """Search hyperparameter space of model for optimal condistions.

    Args:
        data_train: Training dataset
        data_val: Validation dataset
        hp_default: Default values for hyperparameters outside the model structure.

    Returns:
        Best tuned model
    """
    for batch in data_train.take(1):
        inputs, _ = batch
        input_shape = inputs.numpy().shape[1:3]

    tuner = keras_tuner.RandomSearch(
        hypermodel=sh.buildModel(hp_default, input_shape),
        objective="loss",
        max_trials=hp_default["max_trials"],
        executions_per_trial=2,
        overwrite=hp_default["overwrite"],
        directory=hp_default["directory"],
        project_name=hp_default["project"],
    )

    tuner.search(
        data_train,
        validation_data=data_val,
        callbacks=[keras.callbacks.TensorBoard(hp_default["log_path"])],
        verbose=verbosity,
    )

    tuner.results_summary()

    model = tuner.get_best_models()[0]
    return model

import tensorflow as tf


class EarlyStopperCallback(tf.keras.callbacks.Callback):
    """
    Stop the training when a metric is stable for a number of epochs.
    """
    def __init__(self, mse_threshold=None, mae_threshold=None, n_epoch_needed=5):
        """
        Constructor for StopOnStableMetric class.
        :param mse_threshold: The threshold for the Mean Squared Error (MSE) metric
        :param mae_threshold: The threshold for the Mean Absolute Error (MAE) metric
        :param n_epoch_needed: The number of epochs to wait for the metric to be stable
        """
        super(EarlyStopperCallback, self).__init__()
        self.mse_threshold = mse_threshold
        self.mae_threshold = mae_threshold
        self.n_epoch_needed = n_epoch_needed
        self.n_epoch_consecutive = 0
        self.metric_below_threshold = False

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        mse = logs.get('mse')
        mae = logs.get('mae')

        if self.mse_threshold is not None and mse is not None:
            if mse < self.mse_threshold:
                self.metric_below_threshold = True
            else:
                self.metric_below_threshold = False

        if self.mae_threshold is not None and mae is not None:
            if mae < self.mae_threshold:
                self.metric_below_threshold = True
            else:
                self.metric_below_threshold = False

        if self.metric_below_threshold:
            self.n_epoch_consecutive += 1
            print(f"\nEpoch {epoch + 1}: metric under threshold ({self.n_epoch_consecutive}/{self.n_epoch_needed}).")
            if self.n_epoch_consecutive >= self.n_epoch_needed:
                print(f"\nStable metric during {self.n_epoch_needed} epoch. Stopping training.")
                self.model.stop_training = True
        else:
            self.n_epoch_consecutive = 0

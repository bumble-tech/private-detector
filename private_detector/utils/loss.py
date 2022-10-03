"""
Metric module, used for tracking training and eval metrics while training the private detector
"""
from typing import List

import tensorflow as tf


class Metric:
    """
    Base metric class

    Parameters
    ----------
    training : bool
        Whether metric is for training or not
    """
    def __init__(self, training: bool):
        self.training = training
        self.total_loss = tf.keras.metrics.Mean()

        self.ce_loss = tf.keras.metrics.Mean()
        self.reg_loss = tf.keras.metrics.Mean()

        self.acc = tf.keras.metrics.CategoricalAccuracy()

    def reset_states(self) -> None:
        """
        Reset states for all metrics
        """
        self.total_loss.reset_states()

        self.ce_loss.reset_states()
        self.reg_loss.reset_states()

        self.acc.reset_states()


class LossMetricAggregator:
    """
    Module for tracking training and evaluation loss while training the private detector

    Parameters
    ----------
    class_labels : List[str]
        Class labels in training/eval dataset
    global_batch_size : int
        Batch size used in training/eval
    """
    def __init__(self,
                 class_labels: List[str],
                 global_batch_size: int):
        self.global_batch_size = global_batch_size
        self.num_classes = len(class_labels)

        label_smoothing = 0.1

        self.mae = tf.keras.losses.MeanAbsoluteError(
            reduction=tf.keras.losses.Reduction.NONE
        )

        self.ce_loss = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=label_smoothing,
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE,
            name='ce_loss'
        )

        self.train_metric = Metric(training=True)
        self.eval_metric = Metric(training=False)

    def evaluation_result(self) -> float:
        """
        Get accuracy on evaluation set

        Returns
        -------
        accuracy : float
            Accuracy of model on evaluation set
        """
        m = self.eval_metric
        accuracy = m.acc.result()

        return accuracy

    def reset_states(self):
        """
        Reset states of train and evaluation metrics
        """
        self.train_metric.reset_states()
        self.eval_metric.reset_states()

    def loss(self,
             y_true: tf.Tensor,
             y_pred: tf.Tensor,
             training: bool) -> float:
        """
        Calculate categorical crossentropy loss

        Parameters
        ----------
        y_true : tf.Tensor
            True values for dataset
        y_pred : tf.Tensor
            Predicted values from model for dataset
        training : bool
            Whether loss is calculated for training or eval metrics

        Returns
        -------
        ce_loss : float
            Categorical crossentropy loss of predictions vs true labels
        """
        y_pred = tf.cast(
            y_pred,
            tf.float32
        )

        y_true_oh = tf.one_hot(
            y_true,
            self.num_classes
        )

        ce_loss = self.ce_loss(
            y_true_oh,
            y_pred
        )

        m = self.train_metric
        if not training:
            m = self.eval_metric

        m.ce_loss.update_state(ce_loss)
        m.acc.update_state(
            y_true_oh,
            y_pred
        )

        ce_loss = tf.nn.compute_average_loss(
            ce_loss,
            global_batch_size=self.global_batch_size
        )

        return ce_loss

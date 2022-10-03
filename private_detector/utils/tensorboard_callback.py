"""
Tensorboard callback for filling in tensorboard logs after each epoch of the private detector
training loop. Has no effect on the model training itself, just added for logging purposes
"""
from typing import List

import tensorflow as tf


class Callback:
    """
    Callback class to populate tensorboard logs when training the PrivateDetector

    Parameters
    ----------
    log_dir : str
        Logs dir to save tensorboard logs to
    threshold : float
        Threshold above which to consider a probability 'positive'
    """
    def __init__(self,
                 log_dir: str,
                 threshold: float):
        self.loss = tf.keras.metrics.Mean()
        self.accuracy = tf.keras.metrics.Accuracy()
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()
        self.auc = tf.keras.metrics.AUC()
        self.false_negative = tf.keras.metrics.FalseNegatives()
        self.false_positive = tf.keras.metrics.FalsePositives()
        self.true_negative = tf.keras.metrics.TrueNegatives()
        self.true_positive = tf.keras.metrics.TruePositives()

        self.threshold = threshold

        self.summary_writer = tf.summary.create_file_writer(str(log_dir))

    def on_epoch_end(self,
                     loss: float,
                     model: List[int],
                     dataset: tf.Tensor,
                     epoch_num: int):
        """
        What to do at the end of each epoch

        Parameters
        ----------
        loss : float
            Training loss to print to tensorboard
        y_train : List[int]
            Actual values from evaluation data
        predictions : List[int]
            Predictions from model
        epoch_num : int
            Number of current epoch
        """
        for _, images, labels in dataset:
            logits = model(images)
            predictions = tf.nn.softmax(logits, -1)
            binary_preds = predictions.numpy()[:, 1] > self.threshold

            self.loss(loss)
            self.accuracy(labels, binary_preds)
            self.precision(labels, binary_preds)
            self.recall(labels, binary_preds)
            self.auc(labels, binary_preds)
            self.true_positive(labels, binary_preds)
            self.false_positive(labels, binary_preds)
            self.true_negative(labels, binary_preds)
            self.false_negative(labels, binary_preds)

        with self.summary_writer.as_default():
            tf.summary.scalar(
                '0. Loss',
                self.loss.result(),
                step=epoch_num
            )

            tf.summary.scalar(
                '1. Accuracy',
                self.accuracy.result(),
                step=epoch_num
            )

            tf.summary.scalar(
                '2. Precision',
                self.precision.result(),
                step=epoch_num
            )

            tf.summary.scalar(
                '3. Recall',
                self.recall.result(),
                step=epoch_num
            )

            tf.summary.scalar(
                '4. AUC',
                self.auc.result(),
                step=epoch_num
            )

            tf.summary.scalar(
                '5. True Positives',
                self.true_positive.result(),
                step=epoch_num
            )

            tf.summary.scalar(
                '6. True Negatives',
                self.true_negative.result(),
                step=epoch_num
            )

            tf.summary.scalar(
                '7. False Positives',
                self.false_positive.result(),
                step=epoch_num
            )

            tf.summary.scalar(
                '8. False Negatives',
                self.false_negative.result(),
                step=epoch_num
            )

        self.loss.reset_states()
        self.accuracy.reset_states()
        self.precision.reset_states()
        self.recall.reset_states()
        self.auc.reset_states()
        self.true_negative.reset_states()
        self.true_positive.reset_states()
        self.false_negative.reset_states()
        self.false_positive.reset_states()

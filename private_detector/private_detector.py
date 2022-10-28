"""
Private Detector model base class
"""
import re
from pathlib import Path
from typing import List, Tuple

import tensorflow as tf

from .image_dataset import ImageDataset
from .utils import loss
from .utils.efficientnet_config import EfficientNetV2Config
from .utils.effnetv2_model import EffNetV2Model
from .utils.tensorboard_callback import Callback


class PrivateDetector():
    """
    Binary Image Classification module for the PrivateDetector

    Parameters
    ----------
    initial_learning_rate : float
        Initial learning rate for training
    class_labels : List[str]
        Labels for classes in the training set
    checkpoint_dir : str
        Directory to load checkpoints from
    reg_loss_weight : float
        L2 regularization weight
    use_fp16 : bool
        Whether to use float16 or not
    batch_size : int
        Batch size for training
    tensorboard_log_dir : str
        Directory to store tensorboard logs in
    eval_threshold : float
        Threshold above which to consider a probability 'positive' when evaluation
    """
    def __init__(self,
                 initial_learning_rate: float,
                 class_labels: List[str],
                 checkpoint_dir: str,
                 reg_loss_weight: float,
                 use_fp16: bool,
                 batch_size: int,
                 tensorboard_log_dir: str,
                 eval_threshold: float):
        num_classes = len(class_labels)
        config = EfficientNetV2Config(num_classes=num_classes)

        self.model = EffNetV2Model(
            model_name=None,
            model_config=config.model,
            include_top=True
        )

        self.train_image_size = config.train.isize
        self.eval_image_size = config.eval.isize or config.train.isize

        self.epoch_var = tf.Variable(
            0,
            dtype=tf.float32,
            name='epoch_number',
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA
        )

        self.global_step = tf.Variable(
            0,
            dtype=tf.int64,
            name='global_step',
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA
        )

        self.learning_rate = tf.Variable(
            initial_learning_rate,
            dtype=tf.float32,
            name='learning_rate'
        )

        self.initial_learning_rate = initial_learning_rate
        self.opt = tf.optimizers.Adam(learning_rate=self.learning_rate)

        if use_fp16:
            self.opt = tf.keras.mixed_precision.LossScaleOptimizer(
                self.opt,
                dynamic=True
            )

        self.checkpoint = tf.train.Checkpoint(
            step=self.global_step,
            epoch=self.epoch_var,
            model=self.model
        )

        self.manager = tf.train.CheckpointManager(
            self.checkpoint,
            checkpoint_dir,
            max_to_keep=20
        )

        self.good_checkpoint_dir = Path(checkpoint_dir) / 'good'
        self.good_checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.metric = loss.LossMetricAggregator(
            class_labels=class_labels,
            global_batch_size=batch_size
        )

        self.best_metric = 0
        self.reg_loss_weight = reg_loss_weight
        self.use_fp16 = use_fp16
        self.batch_num = None

        self.callback = Callback(
            log_dir=tensorboard_log_dir,
            threshold=eval_threshold
        )

    def fit(self,
            batch_size: int,
            train_dataset: ImageDataset,
            steps_per_train_epoch: int,
            eval_dataset: ImageDataset,
            steps_per_eval_epoch: int,
            reset_on_lr_update: bool,
            min_learning_rate: float,
            num_epochs: int,
            dtype: tf.dtypes.DType,
            skip_saving_epochs: bool,
            epochs_lr_update: int,
            min_eval_metric: float):
        """
        Train model given inputs

        Parameters
        ----------
        batch_size : int
            Batch size to use when training
        train_dataset : ImageDataset
            Dataset of training images
        steps_per_train_epoch : int
            Number of train steps to run per epoch
        eval_dataset : ImageDataset
            Dataset of validation images
        steps_per_eval_epoch : int
            Number of eval steps to run per epoch
        reset_on_lr_update : bool
            Whether to reset to the best model after learning rate update
        min_learning_rate : float
            Minimum learning rate
        num_epochs : int
            Number of epochs to train for
        dtype : tf.dtypes.DType
            Type of variable in the image (e.g. float16)
        skip_saving_epochs : bool
            Do not save good checkpoint and update best metric for this number of the first epochs
        epochs_lr_update : int
            Maximum number of epochs without improvement used to reset or decrease learning rate
        """
        best_saved_path = None
        num_epochs_without_improvement = 0
        initial_learning_rate_multiplier = 0.2
        learning_rate_multiplier = initial_learning_rate_multiplier

        if self.best_metric < min_eval_metric:
            self.best_metric = min_eval_metric

        self.learning_rate.assign(self.initial_learning_rate)

        for epoch_num in range(num_epochs):
            self.metric.reset_states()
            want_reset = False

            _, _, _ = self.run_epoch(
                train_dataset,
                steps_per_train_epoch
            )

            total_loss, true_labels, logits = self.run_epoch(
                eval_dataset,
                steps_per_eval_epoch,
                is_training=False
            )

            self.callback.on_epoch_end(
                loss=total_loss,
                model=self.model,
                dataset=eval_dataset.dataset,
                epoch_num=epoch_num
            )

            self.epoch_var.assign_add(1)

            new_lr = self.learning_rate.numpy()

            new_metric = self.metric.evaluation_result()

            if new_metric > self.best_metric:
                if self.epoch_var.numpy() > skip_saving_epochs:
                    best_saved_path = self.checkpoint.save(
                        file_prefix=f'{self.good_checkpoint_dir}/ckpt-{new_metric:.4f}'
                    )

                    self.best_metric = new_metric

                num_epochs_without_improvement = 0
                self.learning_rate_multiplier = initial_learning_rate_multiplier
            else:
                num_epochs_without_improvement += 1

            if num_epochs_without_improvement >= epochs_lr_update:
                if self.learning_rate > min_learning_rate:
                    new_lr = self.learning_rate.numpy() * learning_rate_multiplier

                    if new_lr < min_learning_rate:
                        new_lr = min_learning_rate

                    if reset_on_lr_update:
                        want_reset = True

                    num_epochs_without_improvement = 0
                    if learning_rate_multiplier > 0.1:
                        learning_rate_multiplier /= 2

                elif num_epochs_without_improvement >= epochs_lr_update:
                    new_lr = self.initial_learning_rate
                    want_reset = True

                    num_epochs_without_improvement = 0
                    learning_rate_multiplier = initial_learning_rate_multiplier

            if want_reset:
                restore_path = tf.train.latest_checkpoint(self.good_checkpoint_dir)

                if restore_path:
                    epoch_num = self.epoch_var.numpy()
                    step_num = self.global_step.numpy()

                    self.checkpoint.restore(best_saved_path)

                    self.epoch_var.assign(epoch_num)
                    self.global_step.assign(step_num)

            # update learning rate even without resetting model
            self.learning_rate.assign(new_lr)

    def restore(self, restore_path: str) -> None:
        """
        Restore model from a checkpoint

        Parameters
        ----------
        restore_path : str
            Path to checkpoint to restore from
        """
        self.checkpoint.restore(restore_path)

    def initial_validation(self,
                           restore_path: str,
                           eval_dataset: ImageDataset,
                           steps_per_eval_epoch: int) -> None:
        """
        Run initial validation with validation set, this is used to change the metrics
        of the model so it's easier to pick up where the training left off

        Parameters
        ----------
        restore_path : str
            Path model was restored from
        eval_dataset : ImageDataset
            Dataset of validation images
        steps_per_eval_epoch : int
            Number of validation steps to run
        """
        self.metric.reset_states()

        self.run_epoch(
            eval_dataset,
            steps_per_eval_epoch,
            False
        )

        self.best_metric = self.metric.evaluation_result()
        self.best_saved_path = restore_path

    def run_epoch(self,
                  dataset: ImageDataset,
                  max_steps: int,
                  is_training: bool = True) -> Tuple[float, tf.Tensor, tf.Tensor]:
        """
        Steps to run for each training epoch

        Parameters
        ----------
        dataset : ImageDataset
            Dataset of images to train/validate on
        max_steps : int
            Max number of steps to take per epoch
        is_training : bool
            Whether this instance is for training or not

        Returns
        -------
        total_loss : float
            Total training loss
        true_labels : tf.Tensor
            Actual y values for dataset
        logits : tf.Tensor
            Logit output of the model for the dataset
        """
        progbar = tf.keras.utils.Progbar(
            target=max_steps,
            stateful_metrics=['loss'],
            unit_name='batch'
        )

        if is_training:

            if tf.distribute.in_cross_replica_context():
                step_func = self.train_step_distributed
            else:
                step_func = self.train_step
        else:
            step_func = self.eval_step

            progbar.verbose = 0

        for batch_num, (_, images, true_labels) in enumerate(dataset):
            total_loss, logits = step_func(
                images=images,
                true_labels=true_labels
            )

            progbar.add(
                n=1,
                values=[('loss', total_loss / (batch_num + 1))]
            )

            if batch_num + 1 >= max_steps:
                break

        return total_loss, true_labels, logits

    @tf.function
    def eval_step(self,
                  images: tf.Tensor,
                  true_labels: tf.Tensor) -> Tuple[float, tf.Tensor]:
        """
        Step to run each time for validation

        Parameters
        ----------
        images : tf.Tensor
            Images to be input into the model
        true_labels : tf.Tensor
            Labels corresponding to those images

        Returns
        -------
        total_loss : float
            Total training loss
        logits : tf.Tensor
            Logit output of the model for the dataset
        """
        logits = self.model(images, training=False)

        ce_loss = self.metric.loss(
            y_true=true_labels,
            y_pred=logits,
            training=False
        )

        total_loss = ce_loss

        self.metric.eval_metric.total_loss.update_state(total_loss)

        return total_loss, logits

    @tf.function
    def train_step_distributed(self,
                               images: tf.Tensor,
                               true_labels: tf.Tensor) -> Tuple[float, tf.Tensor]:
        """
        Step to run each time for training, but distributed

        Parameters
        ----------
        images : tf.Tensor
            Images to be input into the model
        true_labels : tf.Tensor
            Labels corresponding to those images

        Returns
        -------
        total_loss : float
            Total training loss
        logits : tf.Tensor
            Logit output of the model for the dataset
        """
        mirrored_strategy = tf.distribute.get_strategy()

        per_replica_losses, logits = mirrored_strategy.run(
            self.train_step,
            args=(
                images,
                true_labels
            )
        )

        total_loss = mirrored_strategy.reduce(
            tf.distribute.ReduceOp.SUM,
            per_replica_losses,
            axis=None
        )

        return total_loss, logits

    @tf.function
    def train_step(self,
                   images: tf.Tensor,
                   true_labels: tf.Tensor) -> Tuple[float, tf.Tensor]:
        """
        Step to run each time for training

        Parameters
        ----------
        images : tf.Tensor
            Images to be input into the model
        true_labels : tf.Tensor
            Labels corresponding to those images

        Returns
        -------
        total_loss : float
            Total training loss
        logits : tf.Tensor
            Logit output of the model for the dataset
        """
        with tf.GradientTape(persistent=True) as tape:
            logits = self.model(images, training=True)

            ce_loss = self.metric.loss(
                true_labels,
                logits,
                training=True
            )

            total_loss = ce_loss

            if self.reg_loss_weight != 0:
                regex = r'.*(kernel|weight):0$'
                var_match = re.compile(regex)

                l2_loss = self.reg_loss_weight * tf.add_n(
                    [
                        tf.nn.l2_loss(v)
                        for v in self.model.trainable_variables
                        if var_match.match(v.name)
                    ]
                )

                total_loss += l2_loss

            if self.use_fp16:
                scaled_total_loss = self.opt.get_scaled_loss(total_loss)

        if self.reg_loss_weight != 0:
            self.metric.train_metric.reg_loss.update_state(l2_loss)

        self.metric.train_metric.total_loss.update_state(total_loss)
        variables = self.model.trainable_variables

        if self.use_fp16:
            scaled_gradients = tape.gradient(
                scaled_total_loss,
                variables
            )

            gradients = self.opt.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(
                total_loss,
                variables
            )

        gradients, variables = self.return_gradients(
            gradients,
            variables
        )

        self.opt.apply_gradients(zip(gradients, variables))
        self.global_step.assign_add(1)

        return total_loss, logits

    @staticmethod
    def return_gradients(gradients, variables):
        """
        Returns the clipped gradients & variables


        """
        ret_gradients = []
        ret_vars = []

        for g, v in zip(gradients, variables):
            if g is None:
                continue

            g = tf.clip_by_value(g, -10, 10)

            ret_gradients.append(g)
            ret_vars.append(v)

        return ret_gradients, ret_vars

    def save(self, output_dir: str, image_size: int = None) -> None:
        """
        Save model as a SavedModel ready for inference

        Parameters
        ----------
        output_dir : str
            Directory to save model to
        """
        image_size = image_size or self.eval_image_size

        class InferenceModel(tf.keras.Model):
            def __init__(self, model, **kwargs):
                super().__init__(**kwargs)

                self.model = model

            @tf.function(input_signature=[
                tf.TensorSpec(
                    [None, image_size * image_size * 3],
                    tf.float16,
                    name='model_input_images')])
            def __call__(self, inputs):
                images = tf.reshape(
                    inputs,
                    [-1, image_size, image_size, 3]
                )

                logits = self.model(images, False)

                softmax_layer = tf.nn.softmax(logits, -1)

                return softmax_layer

        inference_model = InferenceModel(model=self.model)

        tf.saved_model.save(
            inference_model,
            output_dir
        )

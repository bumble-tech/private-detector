"""
Training script for the private detector
"""
import argparse
from pathlib import Path
from typing import List

import tensorflow as tf
from absl import logging as absl_logging

from private_detector.image_dataset import ImageDataset
from private_detector.private_detector import PrivateDetector
from private_detector.utils.logger import make_logger


def train(train_id: str,
          train_json: List[str],
          eval_json: str,
          num_epochs: int,
          batch_size: int,
          checkpoint_dir: str,
          model_dir: str,
          data_format: str,
          initial_learning_rate: float,
          min_learning_rate: float,
          min_eval_metric: float,
          float_dtype: int,
          steps_per_train_epoch: int,
          steps_per_eval_epoch: int,
          reset_on_lr_update: bool,
          rotation_augmentation: float,
          use_augmentation: str,
          scale_crop_augmentation: float,
          reg_loss_weight: float,
          skip_saving_epochs: int,
          sequential: bool,
          eval_threshold: float,
          epochs_lr_update: int) -> None:
    """
    Train Private Detector model with given parameters

    Parameters
    ----------
    train_id : str
        ID for this particular training run
    train_json : List[str]
        JSON file(s) which describes classes and contains lists of filenames of data files
    eval_json : str
        Validation json file which describes classes and contains lists of filenames of data files
    num_epochs : int
        Number of epochs to train for
    batch_size : int
        Number of images to process in a batch
    checkpoint_dir : str
        Directory to store checkpoints in
    model_dir : str
        Directory to store graph in
    data_format : str
        Data format: [channels_first, channels_last]
    initial_learning_rate : float
        Initial learning rate
    min_learning_rate : float
        Minimal learning rate
    min_eval_metric : float
        Minimal evaluation metric to start saving models
    float_dtype : int
        Float Dtype to use in image tensors
    steps_per_train_epoch : int
        Number of steps per train epoch
    steps_per_eval_epoch : int
        Number of steps per evaluation epoch
    reset_on_lr_update : bool
        Whether to reset to the best model after learning rate update
    rotation_augmentation : float
        Rotation augmentation angle, value <= 0 disables it
    use_augmentation : str
        Add speckle, v0, random or color distortion augmentation
    scale_crop_augmentation : float
        Resize image to the model's size times this scale and then randomly crop needed size
    reg_loss_weight : float
        L2 regularization weight
    skip_saving_epochs : int
        Do not save good checkpoint and update best metric for this number of the first epochs
    sequential : bool
        Use sequential run over randomly shuffled filenames vs equal sampling from each class
    eval_threshold : float
        Threshold above which to consider a prediction positive for evaluation
    epochs_lr_update : int

    Notes
    -----
    Passed as command line arguments: see help documentation in new_train --help
    """
    if checkpoint_dir is None:
        checkpoint_dir = Path(model_dir) / 'checkpoints' / train_id
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    log_dir = Path(model_dir) / 'logs' / train_id
    log_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    logger = make_logger(
        name=train_id,
        directory=log_dir
    )

    mirrored_strategy = tf.distribute.MirroredStrategy()

    if float_dtype == 32:
        dtype = tf.float32
        use_fp16 = False
    elif float_dtype == 16:
        dtype = tf.float16
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        use_fp16 = True

    if scale_crop_augmentation < 1:
        scale_crop_augmentation = 1

    initial_learning_rate *= mirrored_strategy.num_replicas_in_sync

    train_dataset = ImageDataset(
        classes_files=train_json,
        batch_seed=0,
        batch_sequential=sequential,
        batch_size=batch_size,
        steps_per_epoch=steps_per_train_epoch,
        rotation_augmentation=rotation_augmentation,
        use_augmentation=use_augmentation,
        scale_crop_augmentation=scale_crop_augmentation,
        image_dtype=dtype
    )

    logger.info(f"Training dataset loaded from {', '.join(train_json)}")

    eval_dataset = ImageDataset(
        classes_files=eval_json,
        batch_size=batch_size,
        steps_per_epoch=steps_per_eval_epoch,
        rotation_augmentation=rotation_augmentation,
        use_augmentation=use_augmentation,
        scale_crop_augmentation=scale_crop_augmentation,
        image_dtype=dtype,
        is_training=False
    )

    logger.info(f'Evaluation dataset loaded from {eval_json}')

    class_labels = train_dataset.classes
    num_classes = len(class_labels)

    logger.info(f'{num_classes} classes found in dataset: {", ".join(class_labels)}')

    model = PrivateDetector(
        initial_learning_rate=initial_learning_rate,
        class_labels=class_labels,
        checkpoint_dir=checkpoint_dir,
        batch_size=batch_size * mirrored_strategy.num_replicas_in_sync,
        reg_loss_weight=reg_loss_weight,
        use_fp16=use_fp16,
        tensorboard_log_dir=log_dir,
        eval_threshold=eval_threshold
    )

    logger.info('Model initialised')

    restore_path = None

    with mirrored_strategy.scope():
        restore_path = tf.train.latest_checkpoint(checkpoint_dir)

        if restore_path:
            checkpoint_prompt = input(
                f'Checkpoint found at {restore_path}: Continue Training? [y]/n:\n'
            )

            if checkpoint_prompt.lower() not in ['n', 'no', '0']:
                model.restore(restore_path)
                logger.info(
                    f"Restored from good checkpoint {restore_path}, running initial validation")

                model.initial_validation(
                    restore_path=restore_path,
                    eval_dataset=eval_dataset,
                    steps_per_eval_epoch=steps_per_eval_epoch
                )
        else:
            logger.info("Initializing from scratch, no latest checkpoint")

        logger.info('Commencing training')

        model.fit(
            batch_size=batch_size,
            train_dataset=train_dataset,
            steps_per_train_epoch=steps_per_train_epoch,
            eval_dataset=eval_dataset,
            steps_per_eval_epoch=steps_per_eval_epoch,
            reset_on_lr_update=reset_on_lr_update,
            min_learning_rate=min_learning_rate,
            num_epochs=num_epochs,
            dtype=dtype,
            skip_saving_epochs=skip_saving_epochs,
            epochs_lr_update=epochs_lr_update,
            min_eval_metric=min_eval_metric
        )

    logger.info(f'Training complete, saving model to {train_id}')

    model.save(train_id)

    logger.info(f'Model saved to {train_id}')


if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR')
    absl_logging.set_verbosity(absl_logging.ERROR)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train_id',
        type=str,
        required=True,
        help='ID for this particular training run')
    parser.add_argument(
        '--train_json',
        type=str,
        required=True,
        action='append',
        help='JSON file which describes classes and contains lists of filenames of data files')
    parser.add_argument(
        '--eval_json',
        type=str,
        required=True,
        help='Validation JSON file, just like the training file')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Number of images to process in a batch')
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=100,
        help='Number of epochs to train for')
    parser.add_argument(
        '--skip_saving_epochs',
        type=int,
        default=0,
        help='Do not save good checkpoint and update best metric for this many epochs')
    parser.add_argument(
        '--model_dir',
        type=str,
        default='.',
        help='Directory to store graph in')
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        help='Checkpoint directory to load checkpoint from')
    parser.add_argument(
        '--data_format',
        type=str,
        default='channels_last',
        choices=['channels_first', 'channels_last'],
        help='Data format: [channels_first, channels_last]')
    parser.add_argument(
        '--initial_learning_rate',
        default=1e-4,
        type=float,
        help='Initial learning rate')
    parser.add_argument(
        '--min_learning_rate',
        default=1e-6,
        type=float,
        help='Minimal learning rate')
    parser.add_argument(
        '--min_eval_metric',
        default=0.01,
        type=float,
        help='Minimal evaluation metric to start saving models')
    parser.add_argument(
        '--epochs_lr_update',
        default=20,
        type=int,
        help='Maximum number of epochs without improvement used to reset/decrease learning rate')
    parser.add_argument(
        '--float_dtype',
        default=16,
        type=int,
        choices=[16, 32],
        help='Float Dtype to use in image tensors')
    parser.add_argument(
        '--steps_per_train_epoch',
        default=800,
        type=int,
        help='Number of steps per train epoch')
    parser.add_argument(
        '--steps_per_eval_epoch',
        default=1,
        type=int,
        help='Number of steps per evaluation epoch')
    parser.add_argument(
        '--reset_on_lr_update',
        action='store_true',
        help='Whether to reset to the best model after learning rate update')
    parser.add_argument(
        '--rotation_augmentation',
        type=float,
        default=0,
        help='Rotation augmentation angle, value <= 0 disables it')
    parser.add_argument(
        '--use_augmentation',
        type=str,
        help='Add speckle, v0, random or color distortion augmentation')
    parser.add_argument(
        '--scale_crop_augmentation',
        type=float,
        default=1.4,
        help="Resize image to the model's size * this scale and then randomly crop needed size")
    parser.add_argument(
        '--reg_loss_weight',
        type=float,
        default=0,
        help='L2 regularization weight')
    parser.add_argument(
        '--eval_threshold',
        type=float,
        default=0.5,
        help='Threshold above which to consider a prediction positive for evaluation')
    parser.add_argument(
        '--sequential',
        action='store_true',
        help='Sequential run over randomly shuffled filenames vs equal sampling from each class')

    args = parser.parse_args()
    train(**vars(args))

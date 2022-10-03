"""
ImageDataset module for loading images from a classes file
"""
from typing import List, Tuple, Generator, Dict

import tensorflow as tf

from .utils.generator import Generator as PDGenerator
from .utils.preprocess import prepare_image


class ImageDataset():
    """
    Dataset object for images, used for training the PrivateDetector

    Parameters
    ----------
    classes_files : List[str]
        List of files with the image paths per class
    batch_seed : int
        Random seed to use
    batch_sequential : bool
        Use sequential run over randomly shuffled filenames vs equal sampling from each class
    batch_size : int
        Batch size of dataset
    steps_per_epoch : int
        Number of steps to run per epoch
    rotation_augmentation : float
        Rotation augmentation angle, value <= 0 disables it
    use_augmentation : str
        Add speckle, v0, random or color distortion augmentation
    scale_crop_augmentation : float
        Resize image to the model\'s size times this scale and then randomly crop needed size
    image_dtype : tf.dtypes.DType
        Dtype to use for images
    train_image_size : int
        Height/Width of image for training
    eval_image_size : int
        Height/Width of image for evaluation
    is_training : bool
        Whether dataset is training or not
    """
    def __init__(self,
                 classes_files: List[str],
                 batch_seed: int = None,
                 batch_sequential: bool = True,
                 batch_size: int = 24,
                 steps_per_epoch: int = 800,
                 rotation_augmentation: float = 0,
                 use_augmentation: str = None,
                 scale_crop_augmentation: float = 1.4,
                 image_dtype: tf.dtypes.DType = tf.float32,
                 train_image_size: int = 384,
                 eval_image_size: int = 480,
                 is_training: bool = True):
        self.batch_size = batch_size
        self.is_training = is_training

        self.dataset, self.labels = self.generate_dataset(
            classes_files,
            batch_seed,
            batch_sequential,
            steps_per_epoch,
            rotation_augmentation,
            use_augmentation,
            scale_crop_augmentation,
            image_dtype,
            train_image_size,
            eval_image_size
        )

        self.classes = list(self.labels.keys())
        self.steps_per_epoch = steps_per_epoch

    def generate_dataset(self,
                         classes_files: List[str],
                         batch_seed: int,
                         batch_sequential: bool,
                         steps_per_epoch: int,
                         rotation_augmentation: float,
                         use_augmentation: str,
                         scale_crop_augmentation: float,
                         image_dtype: tf.dtypes.DType,
                         train_image_size: int,
                         eval_image_size: int) -> Tuple[tf.data.Dataset, Dict[str, int]]:
        """
        Generate dataset from initalisation parameters

        Parameters
        ----------
        classes_files : List[str]
            List of files with the image paths per class
        batch_seed : int
            Random seed to use
        batch_sequential : bool
            Use sequential run over randomly shuffled filenames vs equal sampling from each class
        steps_per_epoch : int
            Number of steps to run per epoch
        rotation_augmentation : float
            Rotation augmentation angle, value <= 0 disables it
        use_augmentation : str
            Add speckle, v0, random or color distortion augmentation
        scale_crop_augmentation : float
            Resize image to the model\'s size times this scale and then randomly crop needed size
        image_dtype : tf.dtypes.DType
            Dtype to use for images
        train_image_size : int
            Height/Width of image for training
        eval_image_size : int
            Height/Width of image for evaluation

        Returns
        -------
        dataset : tf.data.Dataset
            Image dataset parsed from input file
        labels : Dict[str, int]
            Dictionary of class name : label from input file
        """
        batch_generator = PDGenerator(
            classes_files=classes_files,
            seed=batch_seed,
            sequential=batch_sequential
        )

        labels = batch_generator.labels

        if steps_per_epoch < 0:
            steps_per_epoch = (
                batch_generator.num_images()
                + self.batch_size
                - 1
            ) // (self.batch_size)

        dataset = tf.data.Dataset.from_generator(
            lambda: self.gen(
                batch_generator=batch_generator,
                steps_per_epoch=steps_per_epoch,
            ),
            output_types=(tf.string, tf.int32)
        )

        dataset = dataset.map(
            lambda path, label: self.tf_read_image(
                filename=path,
                label=label,
                train_image_size=train_image_size,
                eval_image_size=eval_image_size,
                is_training=self.is_training,
                rotation_augmentation=rotation_augmentation,
                dtype=image_dtype,
                use_augmentation=use_augmentation,
                scale_crop_augmentation=scale_crop_augmentation
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        dataset = self.dataset_prep(dataset)

        return dataset, labels

    def dataset_prep(self,
                     dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Prepare TF dataset for training or evaluation

        Parameters
        ----------
        dataset : tf.data.Dataset
            Input dataset to be prepared
        """
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE
        )

        if self.is_training:
            dataset = dataset.repeat()

        return dataset

    def __iter__(self):
        """
        Iterator to use dataset with

        Notes
        -----
        Simply inherits the iterator from the TFDataset
        """
        return self.dataset.__iter__()

    @staticmethod
    def tf_read_image(filename: str,
                      label: int,
                      train_image_size: int,
                      eval_image_size: int,
                      is_training: bool,
                      rotation_augmentation: float,
                      dtype: tf.dtypes.DType,
                      use_augmentation: str,
                      scale_crop_augmentation: float) -> Tuple[str, tf.Tensor, int]:
        """
        Read single image from filename and prepare for training

        Parameters
        ----------
        filename : str
            Path to image
        label : int
            Label of image for training
        train_image_size : int
            Height/Width of image for training
        eval_image_size : int
            Height/Width of image for evaluation
        is_training : bool
            Whether the image is training or not
        rotation_augmentation : float
            Rotation augmentation angle, value <= 0 disables it
        dtype : tf.dtypes.DType
            Dtype of the image
        use_augmentation : str
            Add speckle, v0, random or color distortion augmentation
        scale_crop_augmentation : float
            Resize image to the model's size times this scale and then randomly crop needed size

        Returns
        -------
        filename : str
            Path to image
        image : tf.Tensor
            The image itself
        label : int
            The label of the image
        """
        image = tf.io.read_file(filename)
        image = tf.io.decode_jpeg(image, channels=3)

        image = prepare_image(
            image,
            train_image_size,
            eval_image_size,
            is_training,
            rotation_augmentation,
            dtype,
            use_augmentation,
            scale_crop_augmentation
        )

        return filename, image, label

    def gen(self,
            batch_generator: PDGenerator,
            steps_per_epoch: int) -> Generator[Tuple[str, int], None, None]:
        """
        Generate paths and labels using a base Generator of images

        Parameters
        ----------
        batch_generator : Generator
            Base generators to use
        steps_per_epoch : int
            Number of steps to run per epoch

        Returns
        -------
        path : str
            Path to image
        label : int
            Label of image for training
        """
        want_full = not self.is_training

        image_paths = zip(
            *batch_generator.get(
                num=steps_per_epoch * self.batch_size,
                want_full=want_full
            )
        )

        for path, label in image_paths:
            yield path, label

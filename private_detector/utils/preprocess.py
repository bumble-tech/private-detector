"""
Module for preprocessing images used for training the private detector
"""
from typing import Any, Tuple

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.ops import control_flow_ops

from . import autoaugment


def apply_with_random_selector(x: tf.Tensor,
                               func: callable,
                               num_cases: int) -> Any:
    """
    Computes func(x, sel), with sel sampled dynamically from [0...num_cases-1]

    Parameters
    ----------
    x : tf.Tensor
        Input tensor
    func : callable
        Python function to apply
    num_cases : int
        Number of cases to sample sel from

    Returns
    -------
    applied_values : Any
        The result of func(x, sel)
    """
    sel = tf.random.uniform(
        [],
        maxval=num_cases,
        dtype=tf.int32
    )

    # Pass the real x only to one of the func calls.
    applied_values = control_flow_ops.merge([
        func(
            control_flow_ops.switch(
                x,
                tf.equal(sel, case)
            )[1],
            case
        ) for case in range(num_cases)
    ])[0]

    return applied_values


def distort_color(image: tf.Tensor,
                  color_ordering: int = 0,
                  fast_mode: bool = False,
                  scope: str = 'distort_color') -> tf.Tensor:
    """
    Distort the color of a Tensor image.

    Parameters
    ----------
    image : tf.Tensor
        Single image in [0, 1]
    color_ordering : int
        A type of distortion (valid values: 0-3).
    fast_mode : bool
        Avoids slower ops (random_hue and random_contrast)
    scope : str
        Optional scope for name_scope.

    Returns
    -------
    output_image : tf.Tensor
        3-D Tensor color-distorted image on range [0, 1]

    Notes
    -----
    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.
    """

    with tf.name_scope(scope):
        saturation_lower = 0.9
        saturation_upper = 1.2
        brightness_max_delta = 8 / 255
        hue_max_delta = 0.2
        contrast_lower = 0.9
        contrast_upper = 1.2

        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(
                    image,
                    max_delta=brightness_max_delta
                )

                image = tf.image.random_saturation(
                    image,
                    lower=saturation_lower,
                    upper=saturation_upper
                )
            else:
                image = tf.image.random_saturation(
                    image,
                    lower=saturation_lower,
                    upper=saturation_upper
                )

                image = tf.image.random_brightness(
                    image,
                    max_delta=brightness_max_delta
                )
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(
                    image,
                    max_delta=brightness_max_delta
                )

                image = tf.image.random_saturation(
                    image,
                    lower=saturation_lower,
                    upper=saturation_upper
                )

                image = tf.image.random_hue(
                    image,
                    max_delta=hue_max_delta
                )

                image = tf.image.random_contrast(
                    image,
                    lower=contrast_lower,
                    upper=contrast_upper
                )
            elif color_ordering == 1:
                image = tf.image.random_saturation(
                    image,
                    lower=saturation_lower,
                    upper=saturation_upper
                )

                image = tf.image.random_brightness(
                    image,
                    max_delta=brightness_max_delta
                )

                image = tf.image.random_contrast(
                    image,
                    lower=contrast_lower,
                    upper=contrast_upper
                )

                image = tf.image.random_hue(
                    image,
                    max_delta=hue_max_delta
                )
            elif color_ordering == 2:
                image = tf.image.random_contrast(
                    image,
                    lower=contrast_lower,
                    upper=contrast_upper
                )

                image = tf.image.random_hue(
                    image,
                    max_delta=hue_max_delta
                )

                image = tf.image.random_brightness(
                    image,
                    max_delta=brightness_max_delta
                )

                image = tf.image.random_saturation(
                    image,
                    lower=saturation_lower,
                    upper=saturation_upper
                )
            elif color_ordering == 3:
                image = tf.image.random_hue(
                    image,
                    max_delta=hue_max_delta
                )

                image = tf.image.random_saturation(
                    image,
                    lower=saturation_lower,
                    upper=saturation_upper
                )

                image = tf.image.random_contrast(
                    image,
                    lower=contrast_lower,
                    upper=contrast_upper
                )

                image = tf.image.random_brightness(
                    image,
                    max_delta=brightness_max_delta
                )
            else:
                raise ValueError('color_ordering must be in [0, 3]')

        output_image = tf.clip_by_value(image, 0, 1)

        return output_image


def preprocess_for_train(image: tf.Tensor,
                         image_size: int,
                         rotation_augmentation: float,
                         dtype: tf.dtypes.DType,
                         use_augmentation: str,
                         scale_crop_augmentation: float) -> tf.Tensor:
    """
    Preprocess image for training

    Parameters
    ----------
    image : tf.Tensor
        Image to be prepared
    image_size : int
        Height/Width of image for training
    rotation_augmentation : float
        Rotation augmentation angle, value <= 0 disables it
    dtype : tf.dtypes.DType
        Dtype of the image
    use_augmentation : bool
        Add speckle, v0, random or color distortion augmentation
    scale_crop_augmentation : float
        Resize image to the model's size times this scale and then randomly crop needed size

    Returns
    -------
    image : tf.Tensor
        Preprocessed image used for training
    """
    image_size_l = int(image_size * scale_crop_augmentation)
    image_size_l = [image_size_l, image_size_l]

    image = pad_resize_image(
        image,
        image_size_l
    )

    image = tf.cast(image, dtype)

    if use_augmentation and tf.random.uniform([], 0, 1) > 0.5:
        for aug in use_augmentation.split(','):
            if aug == 'speckle':
                image = image + image * tf.random.normal(tf.shape(image))
                image = tf.clip_by_value(image, 0, 255)
            elif aug == 'v0':
                image = tf.cast(image, tf.uint8)
                image = autoaugment.distort_image_with_autoaugment(image, 'v0')
                image = tf.cast(image, dtype)
            elif aug == 'random':
                image = tf.cast(image, tf.uint8)
                randaug_num_layers = 2
                randaug_magnitude = 28

                image = autoaugment.distort_image_with_randaugment(
                    image,
                    randaug_num_layers,
                    randaug_magnitude)
                image = tf.cast(image, dtype)
            elif 'color' in aug:
                # image must be in [0, 1] range for this function
                image /= 255

                if aug == 'color_fast_mode':
                    fast_mode = True
                    num_cases = 2
                else:
                    fast_mode = False
                    num_cases = 4

                image = apply_with_random_selector(
                    image,
                    lambda x, ordering: distort_color(
                        x,
                        ordering,
                        fast_mode=fast_mode),
                    num_cases=num_cases
                )

                image *= 255

    if rotation_augmentation > 0 and tf.random.uniform([], 0, 1) > 0.5:
        angle_min = -rotation_augmentation / 180. * 3.1415
        angle_max = rotation_augmentation / 180. * 3.1415

        angle = tf.random.uniform(
            [],
            minval=angle_min,
            maxval=angle_max,
            dtype=tf.float32
        )

        image = tfa.image.rotate(
            image,
            angle,
            interpolation='BILINEAR'
        )

    if tf.shape(image)[0] != image_size or tf.shape(image)[1] != image_size:
        image = tf.image.random_crop(
            image,
            [image_size, image_size, 3]
        )

    image = tf.image.random_flip_left_right(image)

    image = tf.cast(image, dtype)
    image = tf.clip_by_value(
        image,
        0,
        255
    )

    image -= 128
    image /= 128

    return image


def pad_resize_image(image: tf.Tensor,
                     dims: Tuple[int, int]) -> tf.Tensor:
    """
    Resize image with padding

    Parameters
    ----------
    image : tf.Tensor
        Image to resize
    dims : Tuple[int, int]
        Dimensions of resized image

    Returns
    -------
    image : tf.Tensor
        Resized image
    """
    image = tf.image.resize(
        image,
        dims,
        preserve_aspect_ratio=True
    )

    shape = tf.shape(image)

    sxd = dims[1] - shape[1]
    syd = dims[0] - shape[0]

    sx = tf.cast(
        sxd / 2,
        dtype=tf.int32
    )
    sy = tf.cast(
        syd / 2,
        dtype=tf.int32
    )

    paddings = tf.convert_to_tensor([
        [sy, syd - sy],
        [sx, sxd - sx],
        [0, 0]
    ])

    image = tf.pad(
        image,
        paddings,
        mode='CONSTANT',
        constant_values=128
    )

    return image


def preprocess_for_evaluation(image: tf.Tensor,
                              image_size: int,
                              dtype: tf.dtypes.DType) -> tf.Tensor:
    """
    Preprocess image for evaluation

    Parameters
    ----------
    image : tf.Tensor
        Image to be preprocessed
    image_size : int
        Height/Width of image to be resized to
    dtype : tf.dtypes.DType
        Dtype of image to be used

    Returns
    -------
    image : tf.Tensor
        Image ready for evaluation
    """
    image = pad_resize_image(
        image,
        [image_size, image_size]
    )

    image = tf.cast(image, dtype)

    image -= 128
    image /= 128

    return image


def prepare_image(image: tf.Tensor,
                  train_image_size: int,
                  eval_image_size: int,
                  is_training: bool,
                  rotation_augmentation: float,
                  dtype: tf.dtypes.DType,
                  use_auto_augmentation: str,
                  scale_crop_augmentation: float) -> tf.Tensor:
    """
    Prepare image for training or evaluation

    image : tf.Tensor
        Image to be prepared
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
    use_augmentation : bool
        Add speckle, v0, random or color distortion augmentation
    scale_crop_augmentation : float
        Resize image to the model's size times this scale and then randomly crop needed size

    Returns
    -------
    image : tf.Tensor
        Final image prepared for training or evaluation
    """
    if is_training:
        image = preprocess_for_train(
            image=image,
            image_size=train_image_size,
            rotation_augmentation=rotation_augmentation,
            dtype=dtype,
            use_augmentation=use_auto_augmentation,
            scale_crop_augmentation=scale_crop_augmentation
        )

        return image
    else:
        image = preprocess_for_evaluation(
            image=image,
            image_size=eval_image_size,
            dtype=dtype
        )

        return image

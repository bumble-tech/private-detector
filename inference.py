import argparse
from typing import List

import tensorflow as tf
from absl import logging as absl_logging

from private_detector.utils.preprocess import preprocess_for_evaluation


def read_image(filename: str) -> tf.Tensor:
    """
    Load and preprocess image for inference with the Private Detector

    Parameters
    ----------
    filename : str
        Filename of image

    Returns
    -------
    image : tf.Tensor
        Image ready for inference
    """
    image = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(image, channels=3)

    image = preprocess_for_evaluation(
        image,
        480,
        tf.float16
    )

    image = tf.reshape(image, -1)

    return image


def inference(model: str , image_paths: List[str]) -> None:
    """
    Get predictions with a Private Detector model

    Parameters
    ----------
    model : str
        Path to saved model
    image_paths : List[str]
        Path(s) to image to be predicted on
    """
    model = tf.saved_model.load(model)

    for image_path in image_paths:
        image = read_image(image_path)

        preds = model([image])

        print(f'Probability: {100 * tf.get_static_value(preds[0])[0]:.2f}% - {image_path}')


if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR')
    absl_logging.set_verbosity(absl_logging.ERROR)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Location of SavedModel to load'
    )

    parser.add_argument(
        '--image_paths',
        type=str,
        nargs='+',
        required=True,
        help='Paths to image paths to predict for'
    )

    args = parser.parse_args()
    inference(**vars(args))

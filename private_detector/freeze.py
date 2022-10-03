import tensorflow as tf

import argparse
import sys

import efficientnet

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help='Input checkpoint to convert to frozen graph')
parser.add_argument('--output', type=str, required=True, help='Output path for frozen protobuf file')

def main(argv=None):
    with tf.Graph().as_default() as g:

        raw = tf.placeholder(tf.uint8, shape=[None, 480 * 480 * 3], name='input/images_rgb')
        images = tf.reshape(raw, [-1, 480, 480, 3])
        images = tf.cast(images, tf.float32)

        images = tf.map_fn(tf.image.per_image_standardization, images)

        logits = efficientnet.build_model(images, 2, is_training=False, reuse=False)

        class_probabilities = tf.nn.softmax(logits, name='output/class_probabilities')

        saver = tf.train.Saver()

        with tf.Session(graph=g) as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            saver.restore(sess, 'checkpoints/private_detector/good/')

            output_graph_def = tf.graph_util.convert_variables_to_constants(sess, g.as_graph_def(), ['output/class_probabilities'])

            with tf.gfile.GFile('test_saved_model3', "wb+") as f:
                f.write(output_graph_def.SerializeToString())

if __name__ == '__main__':
    FLAGS = parser.parse_args()

    tf.app.run()


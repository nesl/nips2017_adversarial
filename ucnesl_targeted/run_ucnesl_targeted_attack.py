
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os

import math
import numpy as np
from scipy.misc import imread
from scipy.misc import imsave
import time
import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
import inception_resnet_v2 

slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_float(
    'iter_alpha', 1.0, 'Step size for one iteration.')

tf.flags.DEFINE_integer(
    'num_iter', 10, 'Number of iterations.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS

class Resnet2Model(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False

  def __call__(self, x_input):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
      _, end_points = inception_resnet_v2.inception_resnet_v2(
            x_input, num_classes=self.num_classes, is_training=False)
    self.built = True
    probs = end_points['Predictions']
    # Strip off the extra reshape op at the output
    return probs

def load_target_class(input_dir):
  """Loads target classes."""
  with tf.gfile.Open(os.path.join(input_dir, 'target_class.csv')) as f:
    return {row[0]: int(row[1]) for row in csv.reader(f) if len(row) >= 2}


def load_images(input_dir, batch_shape):
  """Read png images from input directory in batches.

  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

  Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  images = np.zeros(batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  # all_files = tf.gfile.Glob(os.path.join(input_dir, '*.png'))
  # test_files = [all_files[idx] for x in np.random.choice(len(all_files), 200, replace=False)]
  # for filepath in test_files:
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
    with tf.gfile.Open(filepath) as f:
      image = imread(f, mode='RGB').astype(np.float) / 255.0
    # Images for inception classifier are normalized to be in [-1, 1] interval.
    images[idx, :, :, :] = image * 2.0 - 1.0
    filenames.append(os.path.basename(filepath))
    idx += 1
    if idx == batch_size:
      yield filenames, images
      filenames = []
      images = np.zeros(batch_shape)
      idx = 0
  if idx > 0:
    yield filenames, images


def save_images(images, filenames, output_dir):
  """Saves images to the output directory.

  Args:
    images: array with minibatch of images
    filenames: list of filenames without path
      If number of file names in this list less than number of images in
      the minibatch then only first len(filenames) images will be saved.
    output_dir: directory where to save images
  """
  for i, filename in enumerate(filenames):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # so rescale them back to [0, 1].
    with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
      imsave(f, (images[i, :, :, :] + 1.0) * 0.5, format='png')


def main(_):
  # Images for inception classifier are normalized to be in [-1, 1] interval,
  # eps is a difference between pixels so it should be in [0, 2] interval.
  # Renormalizing epsilon from [0, 255] to [0, 2].
    init_start = time.time()
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    alpha = 2.0 * FLAGS.iter_alpha / 255.0
    # eps  = eps * 0.9
    num_iter = FLAGS.num_iter
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001

    tf.logging.set_verbosity(tf.logging.INFO)

    all_images_taget_class = load_target_class(FLAGS.input_dir)

    graph1 = tf.Graph()
    graph2 = tf.Graph()
    graph3 = tf.Graph()

    sess1 = tf.Session(graph=graph1)
    sess2 = tf.Session(graph=graph2)
    sess3 = tf.Session(graph=graph3)

    all_files =  tf.gfile.Glob(os.path.join(FLAGS.input_dir, '*.png'))
    files_cnt = len(all_files)
    num_batches = math.ceil(files_cnt / FLAGS.batch_size)
    allowed_time = (files_cnt * 4.8) # 500 seconds for 100 images.
    init_end = time.time()
    init_time = init_end - init_start
    allowed_time -= init_time
    time_limit_per_batch = allowed_time / num_batches
    # max_time_per_batch = 450
    # time_limit_per_batch = (FLAGS.batch_size * max_time_per_batch / 100.0)

    with graph1.as_default():
        # Prepare graph
        x_input1 = tf.placeholder(tf.float32, shape=batch_shape)
        x_max1 = tf.clip_by_value(x_input1 + eps, -1.0, 1.0)
        x_min1 = tf.clip_by_value(x_input1 - eps, -1.0, 1.0)

        with slim.arg_scope(inception.inception_v3_arg_scope()):
            inception.inception_v3(
                x_input1, num_classes=num_classes, is_training=False)

        x_adv1 = x_input1
        target_class_input1 = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
        one_hot_target_class1 = tf.one_hot(target_class_input1, num_classes)
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            logits1, end_points = inception.inception_v3(
                x_adv1, num_classes=num_classes, is_training=False, reuse=True)
        cross_entropy1 = tf.losses.softmax_cross_entropy(one_hot_target_class1,
                                                      logits1,
                                                      label_smoothing=0.1,
                                                      weights=1.0)
        cross_entropy1 += tf.losses.softmax_cross_entropy(one_hot_target_class1,
                                                       end_points['AuxLogits'],
                                                       label_smoothing=0.1,
                                                       weights=0.4)
        x_next1 = x_adv1 - alpha * tf.sign(tf.gradients(cross_entropy1, x_adv1)[0])
        x_next1 = tf.clip_by_value(x_next1, x_min1, x_max1)
        x_adv1 = x_next1 
        saver1 = tf.train.Saver(slim.get_model_variables())
        sess1.run(tf.global_variables_initializer())
        saver1.restore(sess1, FLAGS.checkpoint_path)


    with graph2.as_default():
        # Prepare graph
        x_input2 = tf.placeholder(tf.float32, shape=batch_shape)
        x_max2 = tf.clip_by_value(x_input2 + eps, -1.0, 1.0)
        x_min2 = tf.clip_by_value(x_input2 - eps, -1.0, 1.0)

        with slim.arg_scope(inception.inception_v3_arg_scope()):
            inception.inception_v3(
              x_input2, num_classes=num_classes, is_training=False)

        x_adv2 = x_input2
        target_class_input2 = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
        one_hot_target_class2 = tf.one_hot(target_class_input2, num_classes)

        with slim.arg_scope(inception.inception_v3_arg_scope()):
            logits2, end_points = inception.inception_v3(
                    x_adv2, num_classes=num_classes, is_training=False, reuse=True)
        cross_entropy2 = tf.losses.softmax_cross_entropy(one_hot_target_class2,
                                                      logits2,
                                                      label_smoothing=0.1,
                                                      weights=1.0)
        cross_entropy2 += tf.losses.softmax_cross_entropy(one_hot_target_class2,
                                                       end_points['AuxLogits'],
                                                       label_smoothing=0.1,
                                                       weights=0.4)
        x_next2 = x_adv2 - alpha * tf.sign(tf.gradients(cross_entropy2, x_adv2)[0])
        x_next2 = tf.clip_by_value(x_next2, x_min2, x_max2)
        x_adv2 = x_next2

        saver2 = tf.train.Saver(slim.get_model_variables())
        sess2.run(tf.global_variables_initializer())
        saver2.restore(sess2, 'adv_inception_v3.ckpt')

    with graph3.as_default():
        # Prepare graph
        x_input3 = tf.placeholder(tf.float32, shape=batch_shape)
        x_max3 = tf.clip_by_value(x_input3 + eps, -1.0, 1.0)
        x_min3 = tf.clip_by_value(x_input3 - eps, -1.0, 1.0)

        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            inception_resnet_v2.inception_resnet_v2(
              x_input3, num_classes=num_classes, is_training=False)

        x_adv3 = x_input3
        target_class_input3 = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
        one_hot_target_class3 = tf.one_hot(target_class_input3, num_classes)

        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits3, end_points = inception_resnet_v2.inception_resnet_v2(
                    x_adv3, num_classes=num_classes, is_training=False, reuse=True)
        cross_entropy3 = tf.losses.softmax_cross_entropy(one_hot_target_class3,
                                                      logits3,
                                                      label_smoothing=0.1,
                                                      weights=1.0)
        cross_entropy3 += tf.losses.softmax_cross_entropy(one_hot_target_class3,
                                                       end_points['AuxLogits'],
                                                       label_smoothing=0.1,
                                                       weights=0.4)
        x_next3 = x_adv3 - alpha * tf.sign(tf.gradients(cross_entropy3, x_adv3)[0])
        x_next3 = tf.clip_by_value(x_next3, x_min3, x_max3)
        x_adv3 = x_next3

        saver3 = tf.train.Saver(slim.get_model_variables())
        sess3.run(tf.global_variables_initializer())
        saver3.restore(sess3, 'ens_adv_inception_resnet_v2.ckpt')
        
    for filenames, images in load_images(FLAGS.input_dir, batch_shape):
	batch_start = time.time() 
        max_max_iter = 200
        target_class_for_batch = ([all_images_taget_class[n] for n in filenames] 
                                  + [0] * (FLAGS.batch_size - len(filenames)))
        input_images = images
        
        for cur_iter in range(max_max_iter):
            adv_images1 = sess1.run(x_adv1,
                              feed_dict={
                                  x_input1: input_images,
                                  target_class_input1: target_class_for_batch
                              })
            adv_images2 = sess2.run(x_adv2, 
                            feed_dict={
                                x_input2: input_images,
                                target_class_input2: target_class_for_batch
                            })
            adv_images3 = sess3.run(x_adv3, 
                            feed_dict={
                                x_input3: input_images,
                                target_class_input3: target_class_for_batch
                            })
            input_images = (adv_images1 + adv_images2 + adv_images3) / 3.0
	    iter_end = time.time()
            cur_iter_time = (iter_end - batch_start)
            pred_next_iter_time = cur_iter_time * (cur_iter+2) / (cur_iter+1)
	    if pred_next_iter_time > time_limit_per_batch:
		break
		
        adv_images = input_images
        save_images(adv_images, filenames, FLAGS.output_dir)


if __name__ == '__main__':
    tf.app.run()

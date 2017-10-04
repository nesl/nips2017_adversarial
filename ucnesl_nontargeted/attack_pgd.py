"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from PIL import Image

import tensorflow as tf
from timeit import default_timer as timer

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

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 20, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS


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
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
    with tf.gfile.Open(filepath) as f:
      image = np.array(Image.open(f).convert('RGB')).astype(np.float) / 255.0
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
      img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
      Image.fromarray(img).save(f, format='PNG')


class PGD_Attack(object):

  def __init__(self, models, x, y, max_epsilon, k, random_start, weights=None):
    self.models = models

    # maximum perturbation
    self.max_epsilon = max_epsilon
    # maximum number of iterations
    self.k = k
    # step per iteration, overstep a bit in total to allow for more exploration
    if random_start:
      self.step_epsilon = (max_epsilon / k)
    else:
      self.step_epsilon = 1.25 * (max_epsilon / k)
    # use a random start
    self.rand = random_start

    # placeholders
    self.x = x
    self.y = y
    label_mask = tf.one_hot(y, 1001, on_value=1.0, off_value=0.0, dtype=tf.float32)

    losses = []

    for model in models:
      # Carlini Wagner loss on the logits
      #correct_logit = tf.reduce_sum(label_mask * model.logits, axis=1)
      #wrong_logit = tf.reduce_max((1 - label_mask) * model.logits - 10000*label_mask, axis=1)
      #loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
      #loss = -correct_logit
      #loss = tf.nn.softmax_cross_entropy_with_logits(logits=model.logits, labels=label_mask)
      loss = -tf.reduce_sum(tf.nn.softmax(model.logits) * label_mask)
      losses.append(loss)
    
    if weights is not None:
      weights = tf.constant(weights)
      avg_loss = tf.reduce_mean(losses * weights)
    else:
      avg_loss = tf.reduce_mean(losses)
    self.grad = tf.gradients(avg_loss, x)[0]
    #self.grads = [tf.gradients(loss, x)[0] for loss in losses]

  def run(self, x_batch, sess):

    # use the model's prediction as the ground truth
    y_batch = sess.run(self.models[0].preds, feed_dict={self.x: x_batch})

    if self.rand:
      # take a small random step to start
      mul = 0.2
      x_adv = x_batch + np.random.uniform(-mul * self.max_epsilon, mul * self.max_epsilon, x_batch.shape)
      x_adv = np.clip(x_adv, -1, 1)
    else:
      x_adv = np.copy(x_batch)

    for i in range(self.k):
      grad = sess.run(self.grad, feed_dict={self.x: x_adv,
                                            self.y: y_batch})
      #grad = np.mean([np.sign(g) for g in grads], axis=0)
      x_adv += self.step_epsilon * np.sign(grad)

      x_adv = np.clip(x_adv, x_batch - self.max_epsilon, x_batch + self.max_epsilon)
      x_adv = np.clip(x_adv, -1, 1) # ensure valid pixel range

    return x_adv

def main(_):
  full_start = timer()
  # Images for inception classifier are normalized to be in [-1, 1] interval,
  # eps is a difference between pixels so it should be in [0, 2] interval.
  # Renormalizing epsilon from [0, 255] to [0, 2].
  eps = 2.0 * FLAGS.max_epsilon / 255.0
  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
  num_classes = 1001

  tf.logging.set_verbosity(tf.logging.INFO)

  with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)
    y_input = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])

    import models

    initialized_vars = set()
    savers = []

    # list of models in our ensemble
    """
    all_models = [models.InceptionResNetV2Model, models.InceptionV3Model, models.ResNetV2_152_Model, models.InceptionV4Model, models.ResNetV1_152_Model,
    models.AdvInceptionV3Model, models.EnsAdvInceptionResNetV2Model] 
    """
    #all_models = [models.InceptionResNetV2Model, models.InceptionV3Model, models.InceptionV4Model, models.AdvInceptionV3Model, models.EnsAdvInceptionResNetV2Model]
    all_models = [models.InceptionV3Model, models.AdvInceptionV3Model, models.EnsAdvInceptionResNetV2Model, models.ResNetV1_152_Model, models.ResNetV2_152_Model]
    #all_models = [models.InceptionV3Model, models.AdvInceptionV3Model, models.EnsAdvInceptionResNetV2Model]
    #all_models = [models.InceptionV3Model, models.AdvInceptionV3Model, models.EnsAdvInceptionResNetV2Model, models.InceptionResNetV2Model, models.InceptionV4Model, models.ResNetV2_152_Model]
    #all_models = [models.InceptionV3Model, models.AdvInceptionV3Model, models.EnsAdvInceptionResNetV2Model] 

    # build all the models and specify the saver
    """
    for i, model in enumerate(all_models):
      all_models[i] = model(num_classes)
      all_models[i](x_input, FLAGS.batch_size)
      all_vars = slim.get_model_variables()
      savers.append(tf.train.Saver(set(all_vars) - initialized_vars))
      initialized_vars = set(all_vars)
    """
    for i, model in enumerate(all_models):
      all_models[i] = model(num_classes)
      all_models[i](x_input, FLAGS.batch_size)
      all_vars = slim.get_model_variables()
      model_vars = [k for k in all_vars if k.name.startswith(all_models[i].ckpt)]
      var_dict = {v.op.name[len(all_models[i].ckpt) + 1:]: v for v in model_vars}
      savers.append(tf.train.Saver(var_dict))
    
    pgd = PGD_Attack(all_models, x_input, y_input, max_epsilon=eps, k=18, random_start=False, weights=[0.35, 0.2, 0.25, 0.1, 0.1])

    # Run computation
    tot_time = 0.0
    processed = 0.0
    with tf.Session() as sess:
      for model, saver in zip(all_models, savers):
        saver.restore(sess, FLAGS.checkpoint_path + '/' + model.ckpt)
      
      print("Initialization done after {} sec".format(timer() - full_start))
      for filenames, images in load_images(FLAGS.input_dir, batch_shape):
        start = timer()
        adv_images = pgd.run(images, sess)

        save_images(adv_images, filenames, FLAGS.output_dir)
        end = timer()
        tot_time += end - start
        processed += FLAGS.batch_size
        #print('{} images processed per second (100 imgs in {} sec)'.format(
        #  processed / tot_time, 100 * tot_time / processed))

      full_end = timer()
      print("DONE: Processed {} images in {} sec".format(processed, full_end - full_start))

if __name__ == '__main__':
  tf.app.run()

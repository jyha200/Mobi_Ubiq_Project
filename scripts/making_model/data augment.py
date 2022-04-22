# -*- coding: utf-8 -*-
"""
TensorFlow data augmentation
for Mobile and ubiquitous computing

@author: INHWI
Reference : 
https://github.com/tensorflow/models/blob/master/research/object_detection/core/preprocessor.py
http://solarisailab.com/archives/2619
"""

import numpy as np
import matplotlib.pyplot as plt
import functools
from PIL import Image
import tensorflow.compat.v1 as tf
import os



"""
shape augmentation function list:
  1. random_horizontal_flip(X)
  2. random_vertical_flip(X)
  3. random_rotation90(X)
  4. random_image_scale(X)
  5. random_pad_image(X)
  6. random_crop_to_aspect_ratio(X)
  7. random_pad_to_aspect_ratio(X)
color augmentation function list:
  1. random_pixel_value_scale(X),
  2. random_rgb_to_gray(X)
  3. random_adjust_brightness(X)
  4. random_adjust_contrast(X)
  5. random_adjust_hue(X),
  6. random_adjust_saturation(X)
  7. random_distort_color(X)
miscellaneous augmentation function list:
  1. random_black_patches(X)
"""

def _random_integer(minval, maxval, seed):
  """Returns a random 0-D tensor between minval and maxval.
  Args:
    minval: minimum value of the random tensor.
    maxval: maximum value of the random tensor.
    seed: random seed.
  Returns:
    A random 0-D tensor between minval and maxval.
  """
  return tf.random_uniform(
      [], minval=minval, maxval=maxval, dtype=tf.int32, seed=seed)

# TODO(mttang): This method is needed because the current
# tf.image.rgb_to_grayscale method does not support quantization. Replace with
# tf.image.rgb_to_grayscale after quantization support is added.
def _rgb_to_grayscale(images, name=None):
  """Converts one or more images from RGB to Grayscale.
  Outputs a tensor of the same `DType` and rank as `images`.  The size of the
  last dimension of the output is 1, containing the Grayscale value of the
  pixels.
  Args:
    images: The RGB tensor to convert. Last dimension must have size 3 and
      should contain RGB values.
    name: A name for the operation (optional).
  Returns:
    The converted grayscale image(s).
  """
  with tf.name_scope(name, 'rgb_to_grayscale', [images]) as name:
    images = tf.convert_to_tensor(images, name='images')
    # Remember original dtype to so we can convert back if needed
    orig_dtype = images.dtype
    flt_image = tf.image.convert_image_dtype(images, tf.float32)

    # Reference for converting between RGB and grayscale.
    # https://en.wikipedia.org/wiki/Luma_%28video%29
    rgb_weights = [0.2989, 0.5870, 0.1140]
    rank_1 = tf.expand_dims(tf.rank(images) - 1, 0)
    gray_float = tf.reduce_sum(
        flt_image * rgb_weights, rank_1, keep_dims=True)
    gray_float.set_shape(images.get_shape()[:-1].concatenate([1]))
    return tf.image.convert_image_dtype(gray_float, orig_dtype, name=name)

def random_horizontal_flip(image,
                           seed=None):
  """Randomly flips the image and detections horizontally.
  The probability of flipping the image is 50%.
  Args:
    image: rank 3 float32 tensor with shape [height, width, channels].
    seed: random seed
  Returns:
    image: image which is the same shape as input image.
  """

  def _flip_image(image):
    # flip image
    image_flipped = tf.image.flip_left_right(image)
    return image_flipped

  with tf.name_scope('RandomHorizontalFlip', values=[image]):
    do_a_flip_random = tf.random_uniform([], seed=seed)
    do_a_flip_random = tf.greater(do_a_flip_random, 0.5)

    # flip image
    image = tf.cond(do_a_flip_random, lambda: _flip_image(image), lambda: image)

    return image

def random_vertical_flip(image,
                         seed=None):
  """Randomly flips the image and detections vertically.
  The probability of flipping the image is 50%.
  Args:
    image: rank 3 float32 tensor with shape [height, width, channels].
    seed: random seed
  Returns:
    image: image which is the same shape as input image.
  """

  def _flip_image(image):
    # flip image
    image_flipped = tf.image.flip_up_down(image)
    return image_flipped

  with tf.name_scope('RandomVerticalFlip', values=[image]):
    do_a_flip_random = tf.random_uniform([], seed=seed)
    do_a_flip_random = tf.greater(do_a_flip_random, 0.5)

    # flip image
    image = tf.cond(do_a_flip_random, lambda: _flip_image(image), lambda: image)

    return image

def random_rotation90(image,
                      seed=None):
  """Randomly rotates the image and detections 90 degrees counter-clockwise.
  The probability of rotating the image is 50%. This can be combined with
  random_horizontal_flip and random_vertical_flip to produce an output with a
  uniform distribution of the eight possible 90 degree rotation / reflection
  combinations.
  Args:
    image: rank 3 float32 tensor with shape [height, width, channels].
    seed: random seed
  Returns:
    image: image which is the same shape as input image.
  """

  def _rot90_image(image):
    # flip image
    image_rotated = tf.image.rot90(image)
    return image_rotated

  with tf.name_scope('RandomRotation90', values=[image]):
    do_a_rot90_random = tf.random_uniform([], seed=seed)
    do_a_rot90_random = tf.greater(do_a_rot90_random, 0.5)

    # flip image
    image = tf.cond(do_a_rot90_random, lambda: _rot90_image(image),
                    lambda: image)

    return image

def random_image_scale(image,
                       min_scale_ratio=0.5,
                       max_scale_ratio=2.0,
                       seed=None):
  """Scales the image size.
  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels].
    min_scale_ratio: minimum scaling ratio.
    max_scale_ratio: maximum scaling ratio.
    seed: random seed.
  Returns:
    image: image which is the same rank as input image.
  """
  with tf.name_scope('RandomImageScale', values=[image]):
    image_shape = tf.shape(image)
    image_height = image_shape[0]
    image_width = image_shape[1]

    size_coef = tf.random_uniform([], minval=min_scale_ratio, maxval=max_scale_ratio, dtype=tf.float32, seed=seed)

    image_newysize = tf.to_int32(
        tf.multiply(tf.to_float(image_height), size_coef))
    image_newxsize = tf.to_int32(
        tf.multiply(tf.to_float(image_width), size_coef))
    image = tf.image.resize_images(
        image, [image_newysize, image_newxsize], align_corners=True)

    return image

def random_pad_image(image,
                     min_image_size=None,
                     max_image_size=None,
                     pad_color=None,
                     seed=None):
  """Randomly pads the image.
  This function randomly pads the image with zeros. The final size of the
  padded image will be between min_image_size and max_image_size.
  if min_image_size is smaller than the input image size, min_image_size will
  be set to the input image size. The same for max_image_size. The input image
  will be located at a uniformly random location inside the padded image.
  The relative location of the boxes to the original image will remain the same.
  Args:
    image: rank 3 float32 tensor containing 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    min_image_size: a tensor of size [min_height, min_width], type tf.int32.
                    If passed as None, will be set to image size
                    [height, width].
    max_image_size: a tensor of size [max_height, max_width], type tf.int32.
                    If passed as None, will be set to twice the
                    image [height * 2, width * 2].
    pad_color: padding color. A rank 1 tensor of [3] with dtype=tf.float32.
               if set as None, it will be set to average color of the input
               image.
    seed: random seed.
  Returns:
    image: Image shape will be [new_height, new_width, channels].
  """
  if pad_color is None:
    pad_color = tf.reduce_mean(image, axis=[0, 1])

  image_shape = tf.shape(image)
  image_height = image_shape[0]
  image_width = image_shape[1]

  if max_image_size is None:
    max_image_size = tf.stack([image_height * 2, image_width * 2])
  max_image_size = tf.maximum(max_image_size,
                              tf.stack([image_height, image_width]))

  if min_image_size is None:
    min_image_size = tf.stack([image_height, image_width])
  min_image_size = tf.maximum(min_image_size,
                              tf.stack([image_height, image_width]))

  target_height = tf.cond(
      max_image_size[0] > min_image_size[0],
      lambda: _random_integer(min_image_size[0], max_image_size[0], seed),
      lambda: max_image_size[0])

  target_width = tf.cond(
      max_image_size[1] > min_image_size[1],
      lambda: _random_integer(min_image_size[1], max_image_size[1], seed),
      lambda: max_image_size[1])

  offset_height = tf.cond(
      target_height > image_height,
      lambda: _random_integer(0, target_height - image_height, seed),
      lambda: tf.constant(0, dtype=tf.int32))

  offset_width = tf.cond(
      target_width > image_width,
      lambda: _random_integer(0, target_width - image_width, seed),
      lambda: tf.constant(0, dtype=tf.int32))

  new_image = tf.image.pad_to_bounding_box(
      image,
      offset_height=offset_height,
      offset_width=offset_width,
      target_height=target_height,
      target_width=target_width)

  # Setting color of the padded pixels
  image_ones = tf.ones_like(image)
  image_ones_padded = tf.image.pad_to_bounding_box(
      image_ones,
      offset_height=offset_height,
      offset_width=offset_width,
      target_height=target_height,
      target_width=target_width)
  image_color_padded = (1.0 - image_ones_padded) * pad_color
  new_image += image_color_padded

  return new_image

def random_crop_to_aspect_ratio(image,
                                aspect_ratio=1.0,
                                seed=None):
  """Randomly crops an image to the specified aspect ratio.
  Randomly crops the a portion of the image such that the crop is of the
  specified aspect ratio, and the crop is as large as possible. If the specified
  aspect ratio is larger than the aspect ratio of the image, this op will
  randomly remove rows from the top and bottom of the image. If the specified
  aspect ratio is less than the aspect ratio of the image, this op will randomly
  remove cols from the left and right of the image. If the specified aspect
  ratio is the same as the aspect ratio of the image, this op will return the
  image.
  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    aspect_ratio: the aspect ratio of cropped image.
    clip_boxes: whether to clip the boxes to the cropped image.
    seed: random seed.
  Returns:
    image: image which is the same rank as input image.
  Raises:
    ValueError: If image is not a 3D tensor.
  """
  if len(image.get_shape()) != 3:
    raise ValueError('Image should be 3D tensor')

  with tf.name_scope('RandomCropToAspectRatio', values=[image]):
    image_shape = tf.shape(image)
    orig_height = image_shape[0]
    orig_width = image_shape[1]
    orig_aspect_ratio = tf.to_float(orig_width) / tf.to_float(orig_height)
    new_aspect_ratio = tf.constant(aspect_ratio, dtype=tf.float32)
    def target_height_fn():
      return tf.to_int32(tf.round(tf.to_float(orig_width) / new_aspect_ratio))

    target_height = tf.cond(orig_aspect_ratio >= new_aspect_ratio,
                            lambda: orig_height, target_height_fn)

    def target_width_fn():
      return tf.to_int32(tf.round(tf.to_float(orig_height) * new_aspect_ratio))

    target_width = tf.cond(orig_aspect_ratio <= new_aspect_ratio,
                           lambda: orig_width, target_width_fn)

    # either offset_height = 0 and offset_width is randomly chosen from
    # [0, offset_width - target_width), or else offset_width = 0 and
    # offset_height is randomly chosen from [0, offset_height - target_height)
    offset_height = _random_integer(0, orig_height - target_height + 1, seed)
    offset_width = _random_integer(0, orig_width - target_width + 1, seed)

    new_image = tf.image.crop_to_bounding_box(
        image, offset_height, offset_width, target_height, target_width)

    return new_image

def random_pad_to_aspect_ratio(image,
                               aspect_ratio=1.0,
                               min_padded_size_ratio=(1.0, 1.0),
                               max_padded_size_ratio=(2.0, 2.0),
                               seed=None):
  """Randomly zero pads an image to the specified aspect ratio.
  Pads the image so that the resulting image will have the specified aspect
  ratio without scaling less than the min_padded_size_ratio or more than the
  max_padded_size_ratio. If the min_padded_size_ratio or max_padded_size_ratio
  is lower than what is possible to maintain the aspect ratio, then this method
  will use the least padding to achieve the specified aspect ratio.
  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    aspect_ratio: aspect ratio of the final image.
    min_padded_size_ratio: min ratio of padded image height and width to the
                           input image's height and width.
    max_padded_size_ratio: max ratio of padded image height and width to the
                           input image's height and width.
    seed: random seed.
  Returns:
    image: image which is the same rank as input image.
  Raises:
    ValueError: If image is not a 3D tensor.
  """
  if len(image.get_shape()) != 3:
    raise ValueError('Image should be 3D tensor')

  with tf.name_scope('RandomPadToAspectRatio', values=[image]):
    image_shape = tf.shape(image)
    image_height = tf.to_float(image_shape[0])
    image_width = tf.to_float(image_shape[1])
    image_aspect_ratio = image_width / image_height
    new_aspect_ratio = tf.constant(aspect_ratio, dtype=tf.float32)
    target_height = tf.cond(
        image_aspect_ratio <= new_aspect_ratio,
        lambda: image_height,
        lambda: image_width / new_aspect_ratio)
    target_width = tf.cond(
        image_aspect_ratio >= new_aspect_ratio,
        lambda: image_width,
        lambda: image_height * new_aspect_ratio)

    min_height = tf.maximum(
        min_padded_size_ratio[0] * image_height, target_height)
    min_width = tf.maximum(
        min_padded_size_ratio[1] * image_width, target_width)
    max_height = tf.maximum(
        max_padded_size_ratio[0] * image_height, target_height)
    max_width = tf.maximum(
        max_padded_size_ratio[1] * image_width, target_width)

    max_scale = tf.minimum(max_height / target_height, max_width / target_width)
    min_scale = tf.minimum(
        max_scale,
        tf.maximum(min_height / target_height, min_width / target_width))

    scale = tf.random_uniform([], min_scale, max_scale, seed=seed)

    target_height = tf.round(scale * target_height)
    target_width = tf.round(scale * target_width)

    new_image = tf.image.pad_to_bounding_box(
        image, 0, 0, tf.to_int32(target_height), tf.to_int32(target_width))

    return new_image

def random_pixel_value_scale(image,
                             minval=0.9,
                             maxval=1.1,
                             seed=None):
  """Scales each value in the pixels of the image.
     This function scales each pixel independent of the other ones.
     For each value in image tensor, draws a random number between
     minval and maxval and multiples the values with them.
  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 255].
    minval: lower ratio of scaling pixel values.
    maxval: upper ratio of scaling pixel values.
    seed: random seed.
  Returns:
    image: image which is the same shape as input image.
  """
  with tf.name_scope('RandomPixelValueScale', values=[image]):
    color_coef = tf.random_uniform(tf.shape(image), minval=minval, maxval=maxval, dtype=tf.float32, seed=seed)
    image = tf.multiply(image, color_coef)
    image = tf.clip_by_value(image, 0.0, 255.0)

    return image

def random_rgb_to_gray(image,
                       probability=0.1,
                       seed=None):
  """Changes the image from RGB to Grayscale with the given probability.
  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 255].
    probability: the probability of returning a grayscale image.
            The probability should be a number between [0, 1].
    seed: random seed.
  Returns:
    image: image which is the same shape as input image.
  """
  def _image_to_gray(image):
    image_gray1 = _rgb_to_grayscale(image)
    image_gray3 = tf.image.grayscale_to_rgb(image_gray1)
    return image_gray3

  with tf.name_scope('RandomRGBtoGray', values=[image]):
    do_gray_random = tf.random_uniform([], seed=seed)

    image = tf.cond(
        tf.greater(do_gray_random, probability), lambda: image,
        lambda: _image_to_gray(image))

    return image

def random_adjust_brightness(image,
                             max_delta=0.2,
                             seed=None):
  """Randomly adjusts brightness.
  Makes sure the output image is still between 0 and 255.
  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 255].
    max_delta: how much to change the brightness. A value between [0, 1).
    seed: random seed.
  Returns:
    image: image which is the same shape as input image.
    boxes: boxes which is the same shape as input boxes.
  """
  with tf.name_scope('RandomAdjustBrightness', values=[image]):
    delta = tf.random_uniform([], -max_delta, max_delta, seed=seed)
    image = tf.image.adjust_brightness(image / 255, delta) * 255
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)

    return image

def random_adjust_contrast(image,
                           min_delta=0.8,
                           max_delta=1.25,
                           seed=None):
  """Randomly adjusts contrast.
  Makes sure the output image is still between 0 and 255.
  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 255].
    min_delta: see max_delta.
    max_delta: how much to change the contrast. Contrast will change with a
               value between min_delta and max_delta. This value will be
               multiplied to the current contrast of the image.
    seed: random seed.
  Returns:
    image: image which is the same shape as input image.
  """
  with tf.name_scope('RandomAdjustContrast', values=[image]):
    contrast_factor = tf.random_uniform([], min_delta, max_delta, seed=seed)
    image = tf.image.adjust_contrast(image / 255, contrast_factor) * 255
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)

    return image

def random_adjust_hue(image,
                      max_delta=0.02,
                      seed=None):
  """Randomly adjusts hue.
  Makes sure the output image is still between 0 and 255.
  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 255].
    max_delta: change hue randomly with a value between 0 and max_delta.
    seed: random seed.
  Returns:
    image: image which is the same shape as input image.
  """
  with tf.name_scope('RandomAdjustHue', values=[image]):
    delta = tf.random_uniform([], -max_delta, max_delta, seed=seed)
    image = tf.image.adjust_hue(image / 255, delta) * 255
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)

    return image

def random_adjust_saturation(image,
                             min_delta=0.8,
                             max_delta=1.25,
                             seed=None):
  """Randomly adjusts saturation.
  Makes sure the output image is still between 0 and 255.
  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 255].
    min_delta: see max_delta.
    max_delta: how much to change the saturation. Saturation will change with a
               value between min_delta and max_delta. This value will be
               multiplied to the current saturation of the image.
    seed: random seed.
  Returns:
    image: image which is the same shape as input image.
  """
  with tf.name_scope('RandomAdjustSaturation', values=[image]):
    saturation_factor = tf.random_uniform([], min_delta, max_delta, seed=seed)
    image = tf.image.adjust_saturation(image / 255, saturation_factor) * 255
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)

    return image

def random_distort_color(image, color_ordering=0):
  """Randomly distorts color.
  Randomly distorts color using a combination of brightness, hue, contrast and
  saturation changes. Makes sure the output image is still between 0 and 255.
  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 255].
    color_ordering: Python int, a type of distortion (valid values: 0, 1).
  Returns:
    image: image which is the same shape as input image.
  Raises:
    ValueError: if color_ordering is not in {0, 1}.
  """
  with tf.name_scope('RandomDistortColor', values=[image]):
    if color_ordering == 0:
      image = random_adjust_brightness(
          image, max_delta=32. / 255.)
      image = random_adjust_saturation(
          image, min_delta=0.5, max_delta=1.5)
      image = random_adjust_hue(
          image, max_delta=0.2)
      image = random_adjust_contrast(
          image, min_delta=0.5, max_delta=1.5)

    elif color_ordering == 1:
      image = random_adjust_brightness(
          image, max_delta=32. / 255.)
      image = random_adjust_contrast(
          image, min_delta=0.5, max_delta=1.5)
      image = random_adjust_saturation(
          image, min_delta=0.5, max_delta=1.5)
      image = random_adjust_hue(
          image, max_delta=0.2)
    else:
      raise ValueError('color_ordering must be in {0, 1}')
    return image

def random_black_patches(image,
                         max_black_patches=10,
                         probability=0.5,
                         size_to_image_ratio=0.1,
                         random_seed=None):
  """Randomly adds some black patches to the image.
  This op adds up to max_black_patches square black patches of a fixed size
  to the image where size is specified via the size_to_image_ratio parameter.
  Args:
    image: rank 3 float32 tensor containing 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    max_black_patches: number of times that the function tries to add a
                       black box to the image.
    probability: at each try, what is the chance of adding a box.
    size_to_image_ratio: Determines the ratio of the size of the black patches
                         to the size of the image.
                         box_size = size_to_image_ratio *
                                    min(image_width, image_height)
    random_seed: random seed.
  Returns:
    image
  """
  def add_black_patch_to_image(image, idx):
    """Function for adding one patch to the image.
    Args:
      image: image
      idx: counter for number of patches that could have been added
    Returns:
      image with a randomly added black box
    """
    image_shape = tf.shape(image)
    image_height = image_shape[0]
    image_width = image_shape[1]
    box_size = tf.to_int32(
        tf.multiply(
            tf.minimum(tf.to_float(image_height), tf.to_float(image_width)),
            size_to_image_ratio))

    normalized_y_min = tf.random_uniform([], minval=0.0, maxval=(1.0 - size_to_image_ratio), seed=random_seed)
    normalized_x_min=tf.random_uniform([], minval=0.0, maxval=(1.0 - size_to_image_ratio), seed=random_seed)

    y_min = tf.to_int32(normalized_y_min * tf.to_float(image_height))
    x_min = tf.to_int32(normalized_x_min * tf.to_float(image_width))
    black_box = tf.ones([box_size, box_size, 3], dtype=tf.float32)
    mask = 1.0 - tf.image.pad_to_bounding_box(black_box, y_min, x_min,
                                              image_height, image_width)
    image = tf.multiply(image, mask)

    return image

  with tf.name_scope('RandomBlackPatchInImage', values=[image]):
    for idx in range(max_black_patches):
      random_prob = tf.random_uniform([], minval=0.0, maxval=1.0, dtype=tf.float32, seed=random_seed)
      image = tf.cond(
          tf.greater(random_prob, probability), lambda: image,
          functools.partial(add_black_patch_to_image, image=image, idx=idx))

    return image

def drawImage(figureName, image):
  plt.figure(num=figureName)
  plt.imshow(image / 255) # 0-1 float normalize
  plt.show()



def saveImage(figureName, image, path, name):
  plt.figure(num=figureName)
  plt.imshow(image / 255) # 0-1 float normalize
  plt.savefig(path + '_' + figureName + '_' + name)


if __name__ == "__main__":
  # draw original image
  image_path = 'data/class1/label2.jpg'
  
  root_path = './data'
  subdirs = os.listdir(root_path)
  for subdir in subdirs:
      for file in os.listdir(root_path + '/' + subdir):
          
          image_path = root_path + '/' + subdir + '/' + file
          path = root_path + '/' + subdir + '/'
         
          test_image = np.asarray(Image.open(image_path), dtype=np.float32)
          
          with tf.Session() as sess:
            X = tf.placeholder(tf.float32, shape = (None, None, 3))
        
          with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
        
            # shape augmentation
            random_horizontal_flip_image, random_vertical_flip_image, random_rotation90_image, random_image_scale_image, _random_pad_image, \
            random_crop_to_aspect_ratio_image, random_pad_to_aspect_ratio_image \
              = sess.run([random_horizontal_flip(X), random_vertical_flip(X), random_rotation90(X), random_image_scale(X), random_pad_image(X), \
                         random_crop_to_aspect_ratio(X), random_pad_to_aspect_ratio(X)], feed_dict={X: test_image})
                 
            
            #random_horizontal_flip_image = Image.fromarray(random_horizontal_flip_image, 'RGB')
            #random_horizontal_flip_image.save('test.jpg')
            
            
            saveImage('random_horizontal_flip_image', random_horizontal_flip_image, path, file)
            #drawImage('random_vertical_flip_image', random_vertical_flip_image)
            saveImage('random_rotation90_image', random_rotation90_image, path, file)
            saveImage('random_image_scale_image', random_image_scale_image, path, file)
            saveImage('random_pad_image', _random_pad_image, path, file)
            saveImage('random_crop_to_aspect_ratio_image', random_crop_to_aspect_ratio_image, path, file)
            saveImage('random_pad_to_aspect_ratio_image', random_pad_to_aspect_ratio_image, path, file)

    # color augmentation
            random_pixel_value_scale_image, random_rgb_to_gray_image, random_adjust_brightness_image, random_adjust_contrast_image, random_adjust_hue_image, \
            random_adjust_saturation_image, random_distort_color_image \
                  = sess.run([random_pixel_value_scale(X), random_rgb_to_gray(X), random_adjust_brightness(X), random_adjust_contrast(X), random_adjust_hue(X), \
                             random_adjust_saturation(X), random_distort_color(X)], feed_dict={X: test_image})

            saveImage('random_pixel_value_scale_image', random_pixel_value_scale_image, path, file)
            saveImage('random_rgb_to_gray_image', random_rgb_to_gray_image, path, file)
            saveImage('random_adjust_brightness_image', random_adjust_brightness_image, path, file)
            saveImage('random_adjust_contrast_image', random_adjust_contrast_image, path, file)
            saveImage('random_adjust_hue_image', random_adjust_hue_image, path, file)
            saveImage('random_adjust_saturation_image', random_adjust_saturation_image, path, file)
            saveImage('random_distort_color_image', random_distort_color_image, path, file)

    # miscellaneous augmentation
            random_black_patches_image = sess.run(random_black_patches(X), feed_dict={X: test_image})
            saveImage('random_black_patches_image', random_black_patches_image, path, file)

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Image pre-processing utilities.
"""
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np

from tensorflow.python.ops import data_flow_ops


def decode_jpeg(image_buffer, scope=None):  # , dtype=tf.float32):
    """Decode a JPEG string into one 3-D float image Tensor.

    Args:
        image_buffer: scalar string Tensor.
        scope: Optional scope for op_scope.
    Returns:
        3-D float Tensor with values ranging from [0, 1).
    """
    # with tf.op_scope([image_buffer], scope, 'decode_jpeg'):
    # with tf.name_scope(scope, 'decode_jpeg', [image_buffer]):
    with tf.name_scope(scope or 'decode_jpeg'):
        # Decode the string as an RGB JPEG.
        # Note that the resulting image contains an unknown height and width
        # that is set dynamically by decode_jpeg. In other words, the height
        # and width of image is unknown at compile-time.
        image = tf.image.decode_jpeg(image_buffer, channels=3,
                                     fancy_upscaling=False,
                                     dct_method='INTEGER_FAST')

        # image = tf.Print(image, [tf.shape(image)], 'Image shape: ')

        return image


def eval_image(image_buffer, frames, thread_id):
    """Get the image for model evaluation."""
    with tf.name_scope('eval_image'):
        image_list = []
        for i in range(frames):
            image_raw = tf.image.decode_jpeg(image_buffer[i], channels=3, dct_method='INTEGER_FAST')
            image = tf.image.convert_image_dtype(image_raw, dtype=tf.float32)
            image_list.append(image)
        image_clip = tf.stack(image_list)
        if not thread_id:
            tf.summary.image(
                'original_image', image_clip[0:1])
        distorted_image = center_crop_and_resize(image_clip)
        distorted_image *= 255
    return distorted_image


def center_crop_and_resize(image_clip, resize_size=[224, 224], scope=None):
    # image_clip: A 4-D tensor of shape [batch, image_height, image_width, depth]
    # return: cropped_images: A 4-D tensor of shape [batch, resize_size[0], resize_size[1], depth]
    with tf.name_scope(scope or 'center_crop_and_resize'):
        initial_width = tf.cast(tf.shape(image_clip)[1], dtype=tf.float32)
        initial_height = tf.cast(tf.shape(image_clip)[2], dtype=tf.float32)
        resize_width = tf.cast(resize_size[0], dtype=tf.float32)
        resize_height = tf.cast(resize_size[1], dtype=tf.float32)
        x1 = 0.5 * (initial_width - resize_width) / initial_width
        y1 = 0.5 * (initial_height - resize_height) / initial_height
        x2 = resize_width / initial_width + x1
        y2 = resize_height / initial_height + y1
        cropped_images = tf.image.crop_and_resize(
            image_clip,
            [[x1, y1, x2, y2] for i in range(image_clip.shape[0])],
            [i for i in range(image_clip.shape[0])],  # batch index
            resize_size
        )
        return cropped_images


def four_corners_crop_and_resize(image_clip, resize_size=[224, 224], scope=None):
    # image_clip: A 4-D tensor of shape [batch, image_height, image_width, depth]
    # return: cropped_images: A 4-D tensor of shape [batch, resize_size[0], resize_size[1], depth]
    with tf.name_scope(scope or 'four_corners_crop_and_resize'):
        initial_width = tf.cast(tf.shape(image_clip)[1], dtype=tf.float32)
        initial_height = tf.cast(tf.shape(image_clip)[2], dtype=tf.float32)
        resize_width = tf.cast(resize_size[0], dtype=tf.float32)
        resize_height = tf.cast(resize_size[1], dtype=tf.float32)

        def crop_center():
            x1 = 0.5 * (initial_width - resize_width) / initial_width
            y1 = 0.5 * (initial_height - resize_height) / initial_height
            x2 = resize_width / initial_width + x1
            y2 = resize_height / initial_height + y1
            cropped_img = tf.image.crop_and_resize(
                image_clip,
                [[x1, y1, x2, y2] for i in range(image_clip.shape[0])],
                [i for i in range(image_clip.shape[0])],  # batch index
                resize_size
            )
            return cropped_img

        def crop_upper_left():
            x1 = 0.0
            y1 = 0.0
            x2 = resize_width / initial_width
            y2 = resize_height / initial_height
            cropped_img = tf.image.crop_and_resize(
                image_clip,
                [[x1, y1, x2, y2] for i in range(image_clip.shape[0])],
                [i for i in range(image_clip.shape[0])],  # batch index
                resize_size
            )
            return cropped_img

        def crop_upper_right():
            x1 = 0.0
            y1 = (initial_height - resize_height) / initial_height
            x2 = resize_width / initial_width
            y2 = 1.0
            cropped_img = tf.image.crop_and_resize(
                image_clip,
                [[x1, y1, x2, y2] for i in range(image_clip.shape[0])],
                [i for i in range(image_clip.shape[0])],  # batch index
                resize_size
            )
            return cropped_img

        def crop_bottom_left():
            x1 = (initial_width - resize_width) / initial_width
            y1 = 0.0
            x2 = 1.0
            y2 = resize_height / initial_height
            cropped_img = tf.image.crop_and_resize(
                image_clip,
                [[x1, y1, x2, y2] for i in range(image_clip.shape[0])],
                [i for i in range(image_clip.shape[0])],  # batch index
                resize_size
            )
            return cropped_img

        def crop_bottom_right():
            x1 = (initial_width - resize_width) / initial_width
            y1 = (initial_height - resize_height) / initial_height
            x2 = 1.0
            y2 = 1.0
            cropped_img = tf.image.crop_and_resize(
                image_clip,
                [[x1, y1, x2, y2] for i in range(image_clip.shape[0])],
                [i for i in range(image_clip.shape[0])],  # batch index
                resize_size
            )
            return cropped_img

        random_uniform = tf.random_uniform([], 0, 1.0)
        cond_crop_upper_left = (tf.less(random_uniform, .2), crop_upper_left)
        cond_crop_upper_right = (tf.less(random_uniform, .4), crop_upper_right)
        cond_crop_bottom_left = (tf.less(random_uniform, .6), crop_bottom_left)
        cond_crop_bottom_right = (tf.less(random_uniform, .8), crop_bottom_right)
        cropped_images = tf.case([cond_crop_upper_left, cond_crop_upper_right, cond_crop_bottom_left, cond_crop_bottom_right], default=crop_center)
        return cropped_images


def horizontal_flip(image_clip, scope=None):
    # image_clip: A 4-D tensor of shape [batch, image_height, image_width, depth]
    # return: cropped_images: A 4-D tensor of shape [batch, image_height, image_width, depth]
    uniform_random = tf.random_uniform([], 0, 1.0)
    mirror_cond = tf.less(uniform_random, .5)
    flipped_images = tf.cond(
        mirror_cond,
        lambda: tf.reverse(image_clip, [2]),
        lambda: image_clip)
    return flipped_images


def image_resize(image_clip, short_size=256):
    # image_clip: A 4-D tensor of shape [batch, image_height, image_width, depth]
    # return: resized_image: A 4-D tensor of shape [batch, image_height/ratio, image_height/ratio, depth]
    initial_width = tf.shape(image_clip)[1]
    initial_height = tf.shape(image_clip)[2]
    min_ = tf.minimum(initial_width, initial_height)
    ratio = tf.to_float(min_) / tf.constant(short_size, dtype=tf.float32)
    new_width = tf.to_int32(tf.to_float(initial_width) / ratio)
    new_height = tf.to_int32(tf.to_float(initial_height) / ratio)
    resized_image = tf.image.resize_images(image_clip, [new_width, new_height])
    return resized_image


def distort_image(image_buffer, frames, time_window, short_size, cropped_size, thread_id=0, scope=None):
    # The image_buffer is driven from output = tf.reshape(features['images'].values, images_shape)
    # return images_clip after scaled/resized/cropped/flip/re-scaled with size: [batch, cropped_size[0], cropped_size[1], channels=3]
    with tf.name_scope(scope or 'distort_image'):
        start = tf.random_uniform(frames.shape, minval=0, maxval=frames - time_window, dtype=tf.int32, seed=None, name=None)
        image_list = []
        for i in range(time_window):
            image_raw = tf.image.decode_jpeg(image_buffer[start + i], channels=3, dct_method='INTEGER_FAST')
            image = tf.image.convert_image_dtype(image_raw, dtype=tf.float32)
            image_list.append(image)

        image_clip = tf.stack(image_list)
        distorted_image = image_resize(image_clip, short_size)
        distorted_image = four_corners_crop_and_resize(distorted_image, cropped_size)
        distorted_image = horizontal_flip(distorted_image)
        if not thread_id:
            tf.summary.image(
                'cropped_resized_image',
                distorted_image[0:1])
        distorted_image = distort_color(distorted_image, thread_id)
        distorted_image *= 255
        if not thread_id:
            tf.summary.image(
                'final_distorted_image',
                distorted_image[0:1])
        return distorted_image


def distort_color(image, thread_id=0, scope=None):
    """Distort the color of the image.

    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.

    Args:
        image: Tensor containing single image.
        thread_id: preprocessing thread ID.
        scope: Optional scope for op_scope.
    Returns:
        color-distorted image
    """
    # with tf.op_scope([image], scope, 'distort_color'):
    # with tf.name_scope(scope, 'distort_color', [image]):
    with tf.name_scope(scope or 'distort_color'):
        color_ordering = thread_id % 2

        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)

        # The random_* ops do not necessarily clamp.
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image


def distort_flow(flow_buffer, frames, time_window, short_size, cropped_size, thread_id=0, scope=None):
    # The image_buffer is driven from output = tf.reshape(features['images'].values, images_shape)
    # return images_clip after scaled/resized/cropped/flip/re-scaled with size: [batch, cropped_size[0], cropped_size[1], channels=3]
    with tf.name_scope(scope or 'distort_flow'):
        flow_x_buffer = flow_buffer[0]
        flow_y_buffer = flow_buffer[1]
        start = tf.random_uniform(frames.shape, minval=0, maxval=frames - time_window, dtype=tf.int32, seed=None, name=None)
        flow_list = []
        for i in range(time_window):
            flow_x_raw = tf.image.decode_jpeg(flow_x_buffer[start + i], channels=3, dct_method='INTEGER_FAST')
            flow_y_raw = tf.image.decode_jpeg(flow_y_buffer[start + i], channels=3, dct_method='INTEGER_FAST')
            flow_x_raw = tf.image.convert_image_dtype(flow_x_raw, dtype=tf.float32)
            flow_y_raw = tf.image.convert_image_dtype(flow_y_raw, dtype=tf.float32)
            flow = tf.stack([flow_x_raw[:, :, 0], flow_y_raw[:, :, 0]], axis=2)
            flow_list.append(flow)  # [height, width, 2]

        flow_clip = tf.stack(flow_list)
        distorted_flow = image_resize(flow_clip, short_size)
        distorted_flow = four_corners_crop_and_resize(distorted_flow, cropped_size)
        distorted_flow = horizontal_flip(distorted_flow)
        if not thread_id:
            tf.summary.image(
                'cropped_resized_flow',
                distorted_flow[0:1, :, :, 0:1])
        distorted_flow *= 255
        if not thread_id:
            tf.summary.image(
                'final_distorted_flow',
                distorted_flow[0:1, :, :, 0:1])
        return distorted_flow


def eval_flow(flow_buffer, frames, thread_id):
    """Get the image for model evaluation."""
    with tf.name_scope('eval_flow'):
        flow_x_buffer = flow_buffer[0]
        flow_y_buffer = flow_buffer[1]
        flow_list = []
        for i in range(frames):
            flow_x_raw = tf.image.decode_jpeg(flow_x_buffer[i], channels=3, dct_method='INTEGER_FAST')
            flow_y_raw = tf.image.decode_jpeg(flow_y_buffer[i], channels=3, dct_method='INTEGER_FAST')
            flow_x_raw = tf.image.convert_image_dtype(flow_x_raw, dtype=tf.float32)
            flow_y_raw = tf.image.convert_image_dtype(flow_y_raw, dtype=tf.float32)
            flow = tf.stack([flow_x_raw[:, :, 0], flow_y_raw[:, :, 0]], axis=2)
            flow_list.append(flow)  # [height, width, 2]
        flow_clip = tf.stack(flow_list)
        if not thread_id:
            tf.summary.image(
                'eval_flow', flow_clip[0:1, :, :, 0:1])
        distorted_image = center_crop_and_resize(flow_clip)
        distorted_image *= 255
    return distorted_image


class ImagePreprocessor(object):
    """Preprocessor for input images."""

    def __init__(self,
                 height,
                 width,
                 batch_size,
                 device_count,
                 data_type,  # RGB or Flow or Audio
                 cropped_size=[224, 224],
                 short_size=256,
                 time_window=10,
                 dtype=tf.float32,
                 train=True,
                 distortions=True,
                 resize_method=None):
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.device_count = device_count
        self.dtype = dtype
        self.train = train
        self.resize_method = resize_method
        self.distortions = distortions
        self.data_type = data_type
        self.time_window = time_window
        self.cropped_size = cropped_size
        self.short_size = short_size
        if self.batch_size % self.device_count != 0:
            raise ValueError(
                ('batch_size must be a multiple of device_count: '
                 'batch_size %d, device_count: %d') %
                (self.batch_size, self.device_count))
        self.batch_size_per_device = self.batch_size // self.device_count

    def preprocess_img(self, image_buffer, frames, thread_id):
        """Preprocessing image_buffer using thread_id."""
        # Note: Width and height of image is known only at runtime.
        if self.train and self.distortions:
            image = distort_image(image_buffer, frames=frames, time_window=self.time_window,
                                  short_size=self.short_size, cropped_size=self.cropped_size, thread_id=thread_id)
        else:
            image = eval_image(image_buffer, frames=frames, thread_id=thread_id)
        # Note: image is now float32 [height,width,3] with range [0, 255]
        # image = tf.cast(image, tf.uint8) # HACK TESTING
        return image

    def preprocess_flow(self, flow_buffer, frames, thread_id):
        """Preprocessing flow_buffer using thread_id."""
        # Note: Width and height of image is known only at runtime.
        if self.train and self.distortions:
            image = distort_flow(flow_buffer, frames=frames, time_window=self.time_window,
                                 short_size=self.short_size, cropped_size=self.cropped_size, thread_id=thread_id)
        else:
            image = eval_flow(flow_buffer, frames=frames, thread_id=thread_id)
        # Note: image is now float32 [height,width,3] with range [0, 255]
        # image = tf.cast(image, tf.uint8) # HACK TESTING
        return image

    def preprocess_audio(self, audio_buffer, frames, thread_id):
        audio = tf.decode_raw(audio_buffer, tf.float32)
        audio /= 256  # from [-256, 256] to [-1.0, 1.0]
        return audio

    def preprocess(self, data_buffer, frames, thread_id=0):
        if self.data_type is 'rgb':
            processed_data = self.preprocess_img(data_buffer, frames, thread_id)
        elif self.data_type is 'flow':
            processed_data = self.preprocess_flow(data_buffer, frames, thread_id)
        elif self.data_type is 'audio':
            processed_data = self.preprocess_audio(data_buffer, frames, thread_id)
        else:
            raise ('data_type error, get: ', self.data_type)
        return processed_data

    def minibatch(self, file_pattern):
        with tf.name_scope('batch_processing'):
            output_data = [[] for i in range(self.device_count)]
            labels = [[] for i in range(self.device_count)]
            record_input = data_flow_ops.RecordInput(
                file_pattern=file_pattern,
                seed=301,
                parallelism=64,
                buffer_size=10000,
                batch_size=self.batch_size,
                name='record_input')
            records = record_input.get_yield_op()
            records = tf.split(records, self.batch_size, 0)
            records = [tf.reshape(record, []) for record in records]
            for i in xrange(self.batch_size):
                value = records[i]
                data_buffer, label_index, _, frames = self.parse_example_proto(value)

                processed_data = self.preprocess(data_buffer, frames)

                device_index = i % self.device_count
                output_data[device_index].append(processed_data)
                labels[device_index].append(label_index)
            label_index_batch = [None] * self.device_count
            for device_index in xrange(self.device_count):
                output_data[device_index] = tf.parallel_stack(output_data[device_index])
                label_index_batch[device_index] = tf.concat(labels[device_index], 0)

                # dynamic_pad=True) # HACK TESTING dynamic_pad=True
                output_data[device_index] = tf.cast(output_data[device_index], self.dtype)
                if self.data_type is 'rgb':
                    depth = 3
                    output_data[device_index] = tf.reshape(
                        output_data[device_index],
                        shape=[self.batch_size_per_device, self.time_window, self.cropped_size[0], self.cropped_size[1], depth])
                    # shape=[self.batch_size_per_device, -1, self.cropped_size[0], self.cropped_size[1], depth])
                elif self.data_type is 'flow':
                    depth = 2
                    output_data[device_index] = tf.reshape(
                        output_data[device_index],
                        shape=[self.batch_size_per_device, self.time_window, self.cropped_size[0], self.cropped_size[1], depth])
                    # shape=[self.batch_size_per_device, -1, self.cropped_size[0], self.cropped_size[1], depth])
                # elif self.data_type is 'audio':
                    # TBD
                else:
                    raise ('data_type error, get: ', self.data_type)
                label_index_batch[device_index] = tf.reshape(
                    label_index_batch[device_index], [self.batch_size_per_device])
                # Display the training images in the visualizer.
                # tf.summary.image('images', images)

            return output_data, label_index_batch

    def parse_example_proto(self, example_serialized):
        # TBD
        # 1. define self.frame_per_clip
        # 2. well define feature's key words
        """Parses an Example proto containing a training example of an image.
        Args:
            example_serialized: scalar Tensor tf.string containing a serialized
                Example protocol buffer.

        Returns:
            image_buffer: Tensor tf.string containing the contents of a JPEG file.
            label: Tensor tf.int32 containing the label.
            bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
                where each coordinate is [0, 1) and the coordinates are arranged as
                [ymin, xmin, ymax, xmax].
            text: Tensor tf.string containing the human-readable label.
        """
        # Dense features in Example proto.
        feature_map = {
            'number_of_frames': tf.FixedLenFeature([], tf.int64),
            'label_index': tf.FixedLenFeature([], tf.int64),
            'label_name': tf.FixedLenFeature([], tf.string),
            'subset': tf.FixedLenFeature([], tf.string),
            # 'resolution': tf.FixedLenFeature([], tf.string),
            'video': tf.FixedLenFeature([], tf.string),
            'images': tf.VarLenFeature(tf.string),
            'flow_x': tf.VarLenFeature(tf.string),
            'flow_y': tf.VarLenFeature(tf.string),
            'audio': tf.FixedLenFeature([], tf.string)
        }

        features = tf.parse_single_example(example_serialized, feature_map)
        frames = tf.cast(features['number_of_frames'], tf.int32)
        images_shape = tf.parallel_stack([frames])

        label = tf.cast(features['label_index'], dtype=tf.int32)

        if self.data_type is 'rgb':
            output = tf.reshape(features['images'].values, images_shape)
        elif self.data_type is 'flow':
            flow_x = tf.reshape(features['flow_x'].values, images_shape)
            flow_y = tf.reshape(features['flow_y'].values, images_shape)
            output = (flow_x, flow_y)
        elif self.data_type is 'audio':
            output = features['audio']
        else:
            raise ('data_type error, get: ', self.data_type)

        return output, label, features['label_name'], frames

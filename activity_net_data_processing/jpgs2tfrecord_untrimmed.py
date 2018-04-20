"""Easily convert RGB video data (e.g. .avi) to the TensorFlow tfrecords file format with the provided 3 color channels.
 Allows to subsequently train a neural network in TensorFlow with the generated tfrecords.
 Due to common hardware/GPU RAM limitations, this implementation allows to limit the number of frames per
 video actually stored in the tfrecords. The code automatically chooses the frame step size such that there is
 an equal separation distribution of the video images. Implementation supports Optical Flow
 (currently OpenCV's calcOpticalFlowFarneback) as an additional 4th channel.
"""

from tensorflow.python.platform import gfile
from tensorflow.python.platform import flags
from tensorflow.python.platform import app
import cv2 as cv2
import numpy as np
import math
import os
import tensorflow as tf
import time
import json
import shutil
import librosa
import csv
import multiprocessing
import subprocess
from functools import partial

FLAGS = flags.FLAGS
flags.DEFINE_integer('n_videos_in_record', 5,
                     'Number of videos stored in one single tfrecord file')
flags.DEFINE_string('image_color_depth', "uint8",
                    'Color depth as string for the images stored in the tfrecord files. '
                    'Has to correspond to the source video color depth. '
                    'Specified as dtype (e.g. uint8 or uint16)')
flags.DEFINE_string('file_suffix', "*.mp4",
                    'defines the video file type, e.g. .mp4')
flags.DEFINE_string('video_source', '../samples', 'Directory with video files')
flags.DEFINE_string('destination', './output_tmp/videos',
                    'Directory for storing tf records')
flags.DEFINE_string('jpg_path', './images_tmp', 'Directory with video files')
flags.DEFINE_string('json_path', './activity_net.v1-3.min.json', 'Directory with json label files')
flags.DEFINE_integer('width_video', 320, 'the width of the videos in pixels')
flags.DEFINE_integer('height_video', 240, 'the height of the videos in pixels')
flags.DEFINE_integer('n_frames_per_video', 5,
                     'specifies the number of frames to be taken from each video')
flags.DEFINE_integer('FPS', 24,
                     'specifies the FPS to be taken from each video')
flags.DEFINE_integer('n_channels', 3,
                     'specifies the number of channels the videos have')
flags.DEFINE_string('video_filenames', None,
                    'specifies the video file names as a list in the case the video paths shall not be determined by the '
                    'script')
flags.DEFINE_string('rtx_name', 'matthzhuang',
                    'rtx_name for accessing hdfs')
flags.DEFINE_string('proj_name', 'VideoAI',
                    'proj_name for accessing hdfs')
flags.DEFINE_string('token', 'e7412d14-f0e8-43a2-a84c-23fa418d5399',
                    'token for accessing hdfs')
flags.DEFINE_string('hdfs_dir_untrimmed', '/user/VideoAI/rextang/activity_net_full/videos/untrimmed/tfrecords',
                    'hdfs_dir_untrimmed for accessing hdfs')
flags.DEFINE_string('hdfs_dir_trimmed', '/user/VideoAI/rextang/activity_net_full/videos/trimmed/tfrecords',
                    'hdfs_dir_trimmed for accessing hdfs')
flags.DEFINE_integer('workers', 16,
                     'Number of workers for multiprocessing')
flags.DEFINE_integer('batch_start', 0,
                     'batch_start')
flags.DEFINE_integer('batch_end', 1,
                     'batch_end')


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def get_chunks(l, n):
    """Yield successive n-sized chunks from l.
    Used to create n sublists from a list l"""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_video_capture(path):
    assert os.path.isfile(path)
    cap = None
    if path:
        cap = cv2.VideoCapture(path)
    return cap


def get_next_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None

    return np.asarray(frame)


def mp4_to_jpgs(video_name, jpg_path, fps):
    filenames = os.path.join(FLAGS.video_source, 'v_' + video_name + '.mp4')
    if not os.path.isfile(filenames):
        filenames = os.path.join(FLAGS.video_source, 'v_' + video_name + '.mkv')
    # assert os.path.isfile(filenames)
    image_path = os.path.join(jpg_path, video_name)
    if not os.path.isdir(image_path):
        os.makedirs(image_path)
    decode_command = 'ffmpeg -i ' + filenames + ' -r ' + str(fps) + ' ' + image_path + '/frame%06d.jpg'
    result = subprocess.Popen(decode_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def clear_path(jpg_path, video_name):
    image_path = os.path.join(jpg_path, video_name)
    shutil.rmtree(image_path, ignore_errors=False, onerror=None)


def upload_tfrecords(tfrecord_dir, video_name, subset, hdfs_dir, delete_flag=False):
    tfrecord_dir_video = os.path.join(tfrecord_dir, video_name)
    tfrecord_list = gfile.Glob(os.path.join(tfrecord_dir_video, '*.tfrecords'))
    for i, file in enumerate(tfrecord_list):
        print(file)
        upload_command = 's_fs -u ' + FLAGS.rtx_name + ' -b ' + FLAGS.proj_name + ' -t ' + FLAGS.token + ' -put ' + file + ' ' + hdfs_dir + '/' + subset
        # print(upload_command)
        result = os.system(upload_command)
    if delete_flag:
        shutil.rmtree(tfrecord_dir_video, ignore_errors=False, onerror=None)


def decode_audio(file_path, sample_per_sec=22050):
    audio, _ = librosa.load(file_path, sr=sample_per_sec, mono=False)  # Extract audio
    pro_audio = preprocess_audio(audio, sample_per_sec)
    return pro_audio


def get_hdfs_files(hdfs_dir):
    check_command = 's_fs -u $RTX_NAME -b $BUS_NAME -t $TOKEN -ls -R ' + hdfs_dir
    r = os.popen(check_command)
    files_list = []
    for line in r:
        line = line.strip('\r\n')
        file_path = line.split(' ')[-1]
        if '.tfrecords' in file_path:
            video_name = file_path.split('/')[-1][:]
            if video_name not in files_list:
                files_list.append(video_name)

    return files_list


def preprocess_audio(raw_audio, sample_per_sec, minimum_seconds=20):
    # Re-scale audio from [-1.0, 1.0] to [-256.0, 256.0]
    # Return audio with size max(sample_per_sec*minimum_seconds, sample_per_sec*ground_truth_second)
    # Select first channel (mono)
    if len(raw_audio.shape) > 1:
        raw_audio = raw_audio[0]

    raw_audio[raw_audio < -1.0] = -1.0
    raw_audio[raw_audio > 1.0] = 1.0

    # Make range [-256, 256]
    raw_audio *= 256.0

    # Make minimum length available
    min_length = sample_per_sec * minimum_seconds
    if min_length > raw_audio.shape[0]:
        raw_audio = np.tile(raw_audio, int(min_length / raw_audio.shape[0]) + 1)

    # Check conditions
    assert len(raw_audio.shape) == 1, "It seems this audio contains two channels, we only need the first channel"
    assert np.max(raw_audio) <= 256, "It seems this audio contains signal that exceeds 256"
    assert np.min(raw_audio) >= -256, "It seems this audio contains signal that exceeds -256"

    # Shape to 1 x DIM x 1 x 1
    # raw_audio = np.reshape(raw_audio, [1, -1, 1, 1])

    return raw_audio.copy()


def extract_flow(images_name):
    # Extract TVL1 flow and encode it as jpg
    num_frames = len(images_name)
    optical_flow = cv2.DualTVL1OpticalFlow_create()

    flow_x_list = []
    flow_y_list = []
    for i_frames in range(num_frames):
        prev_frame = cv2.imread(images_name[max(0, i_frames - 1)])
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        cur_frame = cv2.imread(images_name[i_frames])
        cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
        # Compute TVL1-flow
        prev_frame_gpu = cv2.UMat(prev_frame)
        cur_frame_gpu = cv2.UMat(cur_frame)
        flow_gpu = optical_flow.calc(prev_frame_gpu, cur_frame_gpu, None)
        flow = flow_gpu.get()
        # truncate [-20, 20]
        flow[flow >= 20] = 20
        flow[flow <= -20] = -20
        # scale to [0, 255]
        flow = flow + 20
        flow = flow / 40  # normalize the data to 0 - 1
        flow = 255 * flow  # Now scale by 255
        flow = flow.astype(np.uint8)
        # Encode to jpg
        flow_x_encode = cv2.imencode('.jpg', flow[:, :, 0])[1]
        flow_y_encode = cv2.imencode('.jpg', flow[:, :, 1])[1]
        flow_x_encode = flow_x_encode.tostring()
        flow_y_encode = flow_y_encode.tostring()
        flow_x_list.append(flow_x_encode)
        flow_y_list.append(flow_y_encode)

    return flow_x_list, flow_y_list


def convert_videos_to_tfrecord(source_path, destination_path, jpg_path, json_path,
                               n_videos_in_record=10, n_frames_per_video=5,
                               file_suffix="*.mp4", fps=12,
                               n_channels=3, width=1280, height=720,
                               color_depth="uint8", video_filenames=None):
    """calls sub-functions convert_video_to_numpy and save_numpy_to_tfrecords in order to directly export tfrecords files
    Args:
        source_path: directory where video videos are stored
        destination_path: directory where tfrecords should be stored
        n_videos_in_record: Number of videos stored in one single tfrecord file
        n_frames_per_video: specifies the number of frames to be taken from each
            video
        file_suffix: defines the video file type, e.g. *.mp4
            dense_optical_flow: boolean flag that controls if optical flow should be
            used and added to tfrecords
        n_channels: specifies the number of channels the videos have
        width: the width of the videos in pixels
        height: the height of the videos in pixels
        color_depth: Color depth as string for the images stored in the tfrecord
        files. Has to correspond to the source video color depth. Specified as
            dtype (e.g. uint8 or uint16)
        video_filenames: specify, if the the full paths to the videos can be
            directly be provided. In this case, the source will be ignored.
    """
    activity_index, json_data = _import_ground_truth(json_path)
    if not activity_index:
        raise RuntimeError('No activity_index files found.')
    if not os.path.isdir(destination_path):
        os.makedirs(destination_path)

    database = json_data['database']
    batch_size = FLAGS.workers * 5
    existing_files_trimmed = get_hdfs_files(FLAGS.hdfs_dir_trimmed)
    existing_files_untrimmed = get_hdfs_files(FLAGS.hdfs_dir_untrimmed)
    st = time.time()
    for i in range(FLAGS.batch_start, min(len(list(database)) // batch_size, FLAGS.batch_end)):
        print('Processing {:04d} of {:04d} batches, time/batch: {:.4f}s'.format(i + 1, len(list(database)) // batch_size, time.time() - st))
        st = time.time()
        pool = multiprocessing.Pool(processes=FLAGS.workers)
        pool.map(partial(processing_tfrecord_upload, destination_path=destination_path, database=database,
                         jpg_path=jpg_path, fps=fps, activity_index=activity_index,
                         existing_files_trimmed=existing_files_trimmed, existing_files_untrimmed=existing_files_untrimmed), list(database)[i * batch_size: min(len(list(database)), (i + 1) * batch_size)])
        pool.close()
        pool.join()


def processing_tfrecord_upload(_video_name, destination_path, database, jpg_path, fps, activity_index, existing_files_trimmed, existing_files_untrimmed):
    with open('failed_videos.csv', 'a') as myfile:
        csv_writer = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        frame_data = database[_video_name]  # JSON data for the current video
        # decode mp42jpgs with ffmpeg
        filenames = os.path.join(FLAGS.video_source, 'v_' + _video_name + '.mp4')
        if not os.path.isfile(filenames):
            filenames = os.path.join(FLAGS.video_source, 'v_' + _video_name + '.mkv')
        destination_path_full = destination_path + '_full'
        try:
            assert os.path.isfile(filenames)
            # if _video_name in existing_files:
            #     return
            check_flag = True
            # Check the trimmed files
            trimmed_file_list = try_names(frame_data, _video_name)
            for i in trimmed_file_list:
                if i not in existing_files_trimmed:
                    check_flag = False
            # Check the untrimmed files
            if (_video_name + '.tfrecords') not in existing_files_untrimmed:
                check_flag = False

            if check_flag:  # both trimmed and untrimmed exists
                return
            # if not check_flag:  # both trimmed and untrimmed exists
            #     print(_video_name + '.tfrecords')

            mp4_to_jpgs(_video_name, jpg_path, fps)
            # For processing untrimmed and trimmed in the same time
            image_list, flow_x_list, flow_y_list, audio = extract_data(jpg_path, _video_name, fps)
            save_jpgs_to_tfrecords_untrimmed(image_list, flow_x_list, flow_y_list, audio,
                                             _video_name, frame_data, destination_path_full, jpg_path,
                                             fps, activity_index)
            save_jpgs_to_tfrecords_trimmed(image_list, flow_x_list, flow_y_list, audio,
                                           _video_name, frame_data, destination_path, jpg_path,
                                           fps, activity_index)
            # delete jpgs file
            clear_path(jpg_path, _video_name)
            # upload tfrecord according to different subset
            upload_tfrecords(destination_path_full, _video_name, frame_data['subset'], FLAGS.hdfs_dir_untrimmed, delete_flag=True)
            upload_tfrecords(destination_path, _video_name, frame_data['subset'], FLAGS.hdfs_dir_trimmed, delete_flag=True)
        except:
            csv_writer.writerow([_video_name])


def extract_data(jpg_path, video_name, fps, audio_fps=22050):
    filenames = os.path.join(FLAGS.video_source, 'v_' + video_name + '.mp4')
    if not os.path.isfile(filenames):
        filenames = os.path.join(FLAGS.video_source, 'v_' + video_name + '.mkv')
    audio = decode_audio(filenames, audio_fps)
    images_name = gfile.Glob(os.path.join(jpg_path, video_name, '*.jpg'))  # Read all jpg-name
    images_name.sort()
    image_list = [tf.gfile.FastGFile(images_name[image_count], 'rb').read() for image_count in range(len(images_name))]
    flow_x_list, flow_y_list = extract_flow(images_name)
    return image_list, flow_x_list, flow_y_list, audio


def save_jpgs_to_tfrecords_untrimmed(image_list_in, flow_x_list_in, flow_y_list_in, audio_in, video_name, frame_data, destination_path, jpg_path, fps, activity_index, audio_fps=22050):
    """Converts an entire dataset into x tfrecords where x=videos/fragmentSize.
    Args:
        data: ndarray(uint32) of shape (v,i,h,w,c) with v=number of videos,
        i=number of images, c=number of image channels, h=image height, w=image
        width
        name: filename; data samples type (train|valid|test)
        total_batch_number: indicates the total number of batches
    """

    anno = frame_data['annotations']
    number_of_segment = len(list(anno))
    # audio = decode_audio(os.path.join(FLAGS.video_source, video_name + '.mp4'), audio_fps)

    writer = None
    feature = {}

    if not os.path.isdir(os.path.join(destination_path, video_name)):
        os.makedirs(os.path.join(destination_path, video_name))

    # filename = os.path.join(destination_path, video_name + '.tfrecords')
    filename = os.path.join(destination_path, video_name, video_name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)

    # images_name = gfile.Glob(os.path.join(jpg_path, video_name, '*.jpg'))  # Read all jpg-name
    # images_name.sort()

    info_begin = 0
    info_end = int(np.floor(frame_data['duration']))
    segment_list = []
    label_index_list = []
    label_name_list = []
    if not anno:
        info_label = 'unknown'
        label_name_list.append(str.encode(info_label))
        label_index_list.append(activity_index[info_label])
        segment_list.append(-1.0)
        segment_list.append(-1.0)
        feature['num_segment'] = _int64_feature(-1)  # if no segment, default is -1
    else:
        for seg_i in range(number_of_segment):
            current_json_data = anno[seg_i]
            info_label = current_json_data['label']
            label_name_list.append(str.encode(info_label))
            label_index_list.append(activity_index[info_label])
            segment_list.append(current_json_data['segment'][0])
            segment_list.append(current_json_data['segment'][1])
        feature['num_segment'] = _int64_feature(number_of_segment)

    # image_list = [tf.gfile.FastGFile(images_name[image_count], 'rb').read() for image_count in range(info_begin * fps, min(info_end * fps, len(images_name)))]
    number_of_frames = info_end * fps - info_begin * fps
    current_audio = audio_in[info_begin * audio_fps: min(info_end * audio_fps, audio_in.shape[0])]
    # flow_x_list, flow_y_list = extract_flow(images_name, info_begin, info_end, fps)
    image_list = image_list_in[info_begin * fps: min(info_end * fps, len(image_list_in))]
    flow_x_list, flow_y_list = flow_x_list_in[info_begin * fps: min(info_end * fps, len(image_list_in))], flow_y_list_in[info_begin * fps: min(info_end * fps, len(image_list_in))]
    assert len(flow_x_list) == len(image_list), "The number of RGB not equal to the number of flow"

    current_audio_raw = current_audio.tostring()

    feature['images'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=image_list))
    feature['flow_x'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=flow_x_list))
    feature['flow_y'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=flow_y_list))
    feature['audio'] = _bytes_feature(current_audio_raw)
    feature['number_of_frames'] = _int64_feature(number_of_frames)
    feature['subset'] = _bytes_feature(str.encode(frame_data['subset']))
    feature['resolution'] = _bytes_feature(str.encode(frame_data['resolution']))
    feature['video'] = _bytes_feature(str.encode(video_name))
    feature['duration'] = _bytes_feature(np.asarray(frame_data['duration'], dtype=np.float32).tostring())
    feature['label_index'] = tf.train.Feature(int64_list=tf.train.Int64List(value=label_index_list))
    feature['label_name'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=label_name_list))
    feature['segment'] = _float_feature(segment_list)

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())
    writer.close()


def save_jpgs_to_tfrecords_trimmed(image_list_in, flow_x_list_in, flow_y_list_in, audio_in, video_name, frame_data, destination_path, jpg_path, fps, activity_index, audio_fps=22050):
    """Converts an entire dataset into x tfrecords where x=videos/fragmentSize.
    Args:
        data: ndarray(uint32) of shape (v,i,h,w,c) with v=number of videos,
        i=number of images, c=number of image channels, h=image height, w=image
        width
        name: filename; data samples type (train|valid|test)
        total_batch_number: indicates the total number of batches
    """

    anno = frame_data['annotations']
    number_of_segment = len(list(anno))
    audio = audio_in

    if not anno:  # Missing annotation
        writer = None
        feature = {}

        if not os.path.isdir(os.path.join(destination_path, video_name)):
            os.makedirs(os.path.join(destination_path, video_name))

        # filename = os.path.join(destination_path, video_name + '.tfrecords')
        filename = os.path.join(destination_path, video_name, video_name + '.tfrecords')
        print('Writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)

        images_name = gfile.Glob(os.path.join(jpg_path, video_name, '*.jpg'))  # Read all jpg-name
        images_name.sort()

        info_begin = 0
        info_end = int(np.floor(frame_data['duration']))
        info_label = 'unknown'

        # image_list = [tf.gfile.FastGFile(images_name[image_count], 'rb').read() for image_count in range(info_begin * fps, min(info_end * fps, len(images_name)))]
        image_list = image_list_in[info_begin * fps: min(info_end * fps, len(image_list_in))]
        number_of_frames = info_end * fps - info_begin * fps
        current_audio = audio[info_begin * audio_fps: min(info_end * audio_fps, audio.shape[0])]
        # flow_x_list, flow_y_list = extract_flow(images_name, info_begin, info_end, fps)
        flow_x_list, flow_y_list = flow_x_list_in[info_begin * fps: min(info_end * fps, len(image_list_in))], flow_y_list_in[info_begin * fps: min(info_end * fps, len(image_list_in))]
        assert len(flow_x_list) == len(image_list), "The number of RGB not equal to the number of flow"

        current_audio_raw = current_audio.tostring()

        feature['images'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=image_list))
        feature['flow_x'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=flow_x_list))
        feature['flow_y'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=flow_y_list))
        feature['audio'] = _bytes_feature(current_audio_raw)
        feature['number_of_frames'] = _int64_feature(number_of_frames)
        feature['subset'] = _bytes_feature(str.encode(frame_data['subset']))
        feature['resolution'] = _bytes_feature(str.encode(frame_data['resolution']))
        feature['video'] = _bytes_feature(str.encode(video_name))
        feature['duration'] = _bytes_feature(np.asarray(frame_data['duration'], dtype=np.float32).tostring())
        feature['label_index'] = _int64_feature(activity_index[info_label])
        feature['label_name'] = _bytes_feature(str.encode(info_label))

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
        writer.close()

    else:
        for i_seg in range(number_of_segment):
            current_json_data = anno[i_seg]

            writer = None
            feature = {}

            if not os.path.isdir(os.path.join(destination_path, video_name)):
                os.makedirs(os.path.join(destination_path, video_name))

            # filename = os.path.join(destination_path, video_name + '.tfrecords')
            filename = os.path.join(destination_path, video_name, video_name + '_' +
                                    '{:02d}'.format(i_seg + 1) + '_of_' +
                                    '{:02d}'.format(number_of_segment) + '.tfrecords')
            print('Writing', filename)
            writer = tf.python_io.TFRecordWriter(filename)

            images_name = gfile.Glob(os.path.join(jpg_path, video_name, '*.jpg'))  # Read all jpg-name
            images_name.sort()
            info_begin = int(np.ceil(current_json_data['segment'][0]))
            info_end = int(np.floor(current_json_data['segment'][1]))
            info_label = current_json_data['label']

            # image_list = [tf.gfile.FastGFile(images_name[image_count], 'rb').read() for image_count in range(info_begin * fps, min(info_end * fps, len(images_name)))]
            number_of_frames = info_end * fps - info_begin * fps
            current_audio = audio[info_begin * audio_fps: min(info_end * audio_fps, audio.shape[0])]
            # flow_x_list, flow_y_list = extract_flow(images_name, info_begin, info_end, fps)
            image_list = image_list_in[info_begin * fps: min(info_end * fps, len(image_list_in))]
            flow_x_list, flow_y_list = flow_x_list_in[info_begin * fps: min(info_end * fps, len(image_list_in))], flow_y_list_in[info_begin * fps: min(info_end * fps, len(image_list_in))]
            assert len(flow_x_list) == len(image_list), "The number of RGB not equal to the number of flow"

            current_audio_raw = current_audio.tostring()
            feature['images'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=image_list))
            feature['flow_x'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=flow_x_list))
            feature['flow_y'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=flow_y_list))
            feature['audio'] = _bytes_feature(current_audio_raw)
            feature['number_of_frames'] = _int64_feature(number_of_frames)
            feature['subset'] = _bytes_feature(str.encode(frame_data['subset']))
            feature['resolution'] = _bytes_feature(str.encode(frame_data['resolution']))
            feature['video'] = _bytes_feature(str.encode(video_name))
            feature['duration'] = _bytes_feature(np.asarray(frame_data['duration'], dtype=np.float32).tostring())
            feature['label_index'] = _int64_feature(activity_index[info_label])
            feature['label_name'] = _bytes_feature(str.encode(info_label))
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            writer.close()


def try_names(frame_data, video_name):
    anno = frame_data['annotations']
    number_of_segment = len(list(anno))
    file_list = []
    if not anno:
        filename = (video_name + '.tfrecords')
        file_list.append(filename)
    else:
        for i_seg in range(number_of_segment):
            filename = (video_name + '_' +
                        '{:02d}'.format(i_seg + 1) + '_of_' +
                        '{:02d}'.format(number_of_segment) + '.tfrecords')
            file_list.append(filename)
    return file_list


def _import_ground_truth(ground_truth_filename):
    """Reads ground truth file, checks if it is well formatted, and returns
       the ground truth instances and the activity classes.
    Parameters
    ----------
    ground_truth_filename : str
        Full path to the ground truth json file.
    Outputs
    -------
    activity_index : dict
        Dictionary containing class index.
    """
    with open(ground_truth_filename, 'r') as fobj:
        data = json.load(fobj)

    # Initialize data frame
    activity_index, cidx = {}, 0
    video_lst, label_lst = [], []
    for videoid, v in data['database'].items():
        for ann in v['annotations']:
            if ann['label'] not in activity_index:
                activity_index[ann['label']] = cidx
                cidx += 1
            video_lst.append(videoid)
            label_lst.append(activity_index[ann['label']])
    activity_index['unknown'] = cidx
    return activity_index, data


def main(argv):
    convert_videos_to_tfrecord(FLAGS.video_source, FLAGS.destination, FLAGS.jpg_path, FLAGS.json_path,
                               FLAGS.n_videos_in_record,
                               FLAGS.n_frames_per_video, FLAGS.file_suffix,
                               FLAGS.FPS, FLAGS.n_channels,
                               FLAGS.width_video, FLAGS.height_video,
                               FLAGS.image_color_depth, FLAGS.video_filenames)


if __name__ == '__main__':
    app.run()

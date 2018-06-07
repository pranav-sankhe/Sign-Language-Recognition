import tensorflow as tf
import cv2
import numpy as np
NUM_EPOCHS = 1
SEQ_NUM_FRAMES = 291


# def decode(serialized_example, sess):
#   # Prepare feature list; read encoded JPG images as bytes
#   features = dict()
#   features["class_label"] = tf.FixedLenFeature((), tf.int64)
#   features["frames"] = tf.VarLenFeature((), tf.string)
#   features["num_frames"] = tf.FixedLenFeature((), tf.int64)

#   # Parse into tensors
#   parsed_features = tf.parse_single_example(serialized_example, features)

#   # Randomly sample offset from the valid range.
#   random_offset = tf.random_uniform(
#       shape=(), minval=0,
#       maxval=parsed_features["num_frames"] - SEQ_NUM_FRAMES, dtype=tf.int64)

#   offsets = tf.range(offset, offset + SEQ_NUM_FRAMES)

#   # Decode the encoded JPG images
#   images = tf.map_fn(lambda i: tf.image.decode_jpeg(parsed_features["frames"][i]),
#                      offsets)

#   label  = tf.cast(parsed_features["class_label"], tf.int64)

#   return images, label


# def video_left_right_flip(images):
#     '''
#     Performs tf.image.flip_left_right on entire list of video frames.
#     Work around since the random selection must be consistent for entire video
#     :param images: Tensor constaining video frames (N,H,W,3)
#     :return: images: Tensor constaining video frames left-right flipped (N,H,W,3)
#     '''
#     images_list = tf.unstack(images)
#     for i in range(len(images_list)):
#         images_list[i] = tf.image.flip_left_right(images_list[i])
#     return tf.stack(images_list)

# def preprocess_video(images, label):
#     '''
#     Given the 'images' Tensor of video frames (N,H,W,3) perform the following
#     preprocessing steps:
#     1. Takes a random crop of size CROP_SIZExCROP_SIZE from the video frames.
#     2. Optionally performs random left-right flipping of the video.
#     3. Performs video normalization, to the range [-0.5, +0.5]
#     :param images: Tensor (tf.uint8) constaining video frames (N,H,W,3)
#     :param label:  Tensor (tf.int64) constaining video frames ()
#     :return:
#     '''

#     # Take a random crop of the video, returns tensor of shape (N,CROP_SIZE,CROP_SIZE,3)
#     images = tf.random_crop(images, (SEQ_NUM_FRAMES, CROP_SIZE, CROP_SIZE, 3))

#     if RANDOM_LEFT_RIGHT_FLIP:
#         # Consistent left_right_flip for entire video
#         sample = tf.random_uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)
#         option = tf.less(sample, 0.5)
#         images = tf.cond(option,
#                          lambda: video_left_right_flip(images),
#                          lambda: tf.identity(images))

#     # Normalization: [0, 255] => [-0.5, +0.5] floats
#     images = tf.cast(images, tf.float32) * (1./255.) - 0.5
#     return images, label

BATCH_SIZE = 1
def decode(serialized_example, sess):
  # Prepare feature list; read encoded JPG images as bytes
  features = dict()
  features["class_label"] = tf.FixedLenFeature((), tf.int64)
  features["frames"] = tf.VarLenFeature((), tf.string)
  features["num_frames"] = tf.FixedLenFeature((), tf.int64)

  # Parse into tensors
  parsed_features = tf.parse_single_example(serialized_example, features)

  # Randomly sample offset from the valid range.
  random_offset = tf.random_uniform(
      shape=(), minval=0,
      maxval=parsed_features["num_frames"] - SEQ_NUM_FRAMES, dtype=tf.int64)

  offsets = tf.range(offset, offset + SEQ_NUM_FRAMES)

  # Decode the encoded JPG images
  images = tf.map_fn(lambda i: tf.image.decode_jpeg(parsed_features["frames"][i]),
                     offsets)

 # label  = tf.cast(parsed_features["class_label"], tf.int64)

  return images#, label




def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_list_feature(values):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
with tf.python_io.TFRecordWriter('tftest') as writer:

  # Read and resize all video frames, np.uint8 of size [N,H,W,3]
  frames = np.load('./npVideos/RG0_Corpus_201801_P150_02_t01.mp4.npy')

  features = {}
  features['num_frames']  = _int64_feature(frames.shape[0])
  features['height']      = _int64_feature(frames.shape[1])
  features['width']       = _int64_feature(frames.shape[2])
  features['channels']    = _int64_feature(frames.shape[3])
#  features['class_label'] = _int64_feature(example['class_id'])
#  features['class_text']  = _bytes_feature(tf.compat.as_bytes(example['class_label']))
#  features['filename']    = _bytes_feature(tf.compat.as_bytes(example['video_id']))

  # Compress the frames using JPG and store in as a list of strings in 'frames'
  encoded_frames = [tf.compat.as_bytes(cv2.imencode(".jpg", frame)[1].tobytes())
                    for frame in frames]
  features['frames'] = _bytes_list_feature(encoded_frames)

  tfrecord_example = tf.train.Example(features=tf.train.Features(feature=features))
  writer.write(tfrecord_example.SerializeToString())


if  __name__ == "__main__":

    import glob
    tfrecord_files = glob.glob("tftest")
    tfrecord_files.sort()

    sess = tf.Session()
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer())

    dataset = tf.data.TFRecordDataset(tfrecord_files)

    dataset = dataset.repeat(NUM_EPOCHS)
    temp = decode(dataset, sess)
    dataset = dataset.map(temp)
    dataset = dataset.map(preprocess_video)

    # The parameter is the queue size
    dataset = dataset.shuffle(1000 + 3 * BATCH_SIZE)
    dataset = dataset.batch(BATCH_SIZE)

    iterator = dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()

    sess.run(init_op)

    while True:

        # Fetch a new batch from the dataset
        batch_videos, batch_labels = sess.run(next_batch)

        for sample_idx in range(BATCH_SIZE):
            print("Class label = {}".format(batch_labels[sample_idx]))
            for frame_idx in range(SEQ_NUM_FRAMES):
                cv2.imshow("image", batch_videos[sample_idx,frame_idx])
                cv2.waitKey(20)
            key = cv2.waitKey(0)
            if key == ord('q'):
                exit()





# import cv2 
# import numpy as np 


# cap = cv2.VideoCapture('test.mp4')
# count = 0 
# print cap
# while(cap.isOpened() == True):

    
#     ret, frame = cap.read()
#     if ret == False:
#         print "video ended or not detected"
#         break
#     frame = cv2.resize(frame, (256, 256))
#     frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     count = str(count)
#     num = '' 
#     diff = 3 - len(count)
#     if diff == 0:
#         pass
#     else:
#         for i in range(diff):
#             num = num + '0'
    
#     num = num + count   
#     if int(count)%2 == 0:
#         cv2.imwrite('test' + '/' + num + '.jpg', frame) 
#     count = int(count)
#     count = count + 1    

























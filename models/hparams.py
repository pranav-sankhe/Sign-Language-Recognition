import tensorflow as tf

MAX_SENT_LENGTH = 16
NUM_SEGMENTS = 9

EMBEDDING_SIZE = 256
VOCAB_SIZE = 279
FC_SIZE = 256 #256
SOS = '<s>'
EOS = '</s>'

BATCH_SIZE = 6
IMG_HEIGHT = 256
IMG_WIDTH = 256
IN_CHANNELS = 2
prescaler = 4
NUM_FRAMES = 858/prescaler


NUM_LSTM_CELLS = 128
NUM_ITERATIONS = 1000

beam_width = 2

mode = 'train'
init_std = 0.01
optimizer = "sgd"
LR = 0.01
max_gradient_norm = 5
TIME_MAJOR = False

DTYPE = tf.float32


_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_VERSION = 2
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES

block_sizes = [2, 2]
# block_sizes = [3, 4, 6, 3]
# block_sizes = [3, 4, 6, 3]
# block_sizes = [3, 4, 23, 3]
# block_sizes = [3, 8, 36, 3]
# block_sizes = [3, 24, 36, 3]
block_strides = [1, 1]
resnet_num_filters = 4
resnet_kernel_size = 3
resnet_conv_stride = 1
resnet_training_bool = True
resnet_first_pool_size = None
resnet_first_pool_stride = 1
bottleneck_bool = True
resnet_version = 2
pre_activation = False


BASE_VIDEO_FILE = '/home/psankhe/sign-lang-recog/data/opflow_xy'
BASE_ANNOT_FILE = '/home/psankhe/sign-lang-recog/annotations'

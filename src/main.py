import cv2
import numpy as np
from PIL import Image
import time
import keras
import tensorflow as tf
from scipy import optimize
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model
from matplotlib import pyplot as plt
from tensorflow.python.client import device_lib
from keras import backend

print(device_lib.list_local_devices())


CONTENT_LAYER = ('block2_conv2')
STYLE_LAYERS = ['block1_conv2', 'block2_conv2',
                  'block3_conv3', 'block4_conv3',
                  'block5_conv3']

CONTENT_WEIGHT = 0.025
STYLE_WEIGHT = 0.2
TV_WEIGHT = 1.0
BATCH_SIZE = 10
BATCHES = 8


def get_frames(video_path, height, width):
    capture = cv2.VideoCapture(video_path)
    success,image = capture.read()
    images = []
    count = 0
    while success:
        cv2.imwrite("../frames/frame%d.jpg" % count,image)
        images.append(cv2.resize(image, (height, width)))
        success, image = capture.read()
        count += 1
    return images

def load_img(img_path):
    img = image.load_img(img_path, target_size=(height, width))
    return img


def preprocess(img, exp):
    x = image.img_to_array(img)
    if exp:
        x = np.expand_dims(x, axis=0)
    # new dims for number in model
    x = preprocess_input(x)
    return x

"""

        inputs shape = (number of frames, height, width, channels)

"""

height = 255
width = 255

# make frames from content video
frames = np.array(get_frames('../video/Barry.mp4',height,width))
frames_count = 0
temp_frames = []
for frame in frames:
    temp_frames.append(preprocess(frame, False))
    frames_count += 1
content_frames = tf.Variable(np.asarray(temp_frames), dtype=tf.float32)

# have to make placehodler with no knowing one dimension of frames to concet inputs
content_ph = tf.placeholder(tf.float32, shape=(None, height, width, 3))

# load style image
style_path = '../style/gog.jpg'
style = preprocess(load_img(style_path), True)
#style = np.expand_dims(style, axis=0)  # make another axis for frames index and to concat with other variables
style_image = tf.Variable(style)


# make placeholder for our target new frames
target_frames = tf.placeholder(tf.float32, shape=(frames_count, height, width, 3))

target_ph = tf.placeholder(tf.float32, shape=(None, height, width, 3))

print(content_ph)
print(style_image)
print(target_ph)

# inputs for CNN
"""

        input tensor:
            [0:frames_count - 1] - Content video
            [frames_count] - Style image
            [frames_count + 1]::] - Target frames
            
        
"""
input_tensor = tf.concat([content_ph,
                          style_image,
                          target_ph], axis=0)
print(input_tensor)


# create model

# input only allow 4 dim input tensor, other ones are incompatible with layers

model = VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False,)

layers = dict([(layer.name, layer.output) for layer in model.layers])



# Content loss


def content_loss(content_weight , content, combination):
    return content_weight * tf.reduce_sum(tf.square(combination - content))

c_loss = []

# loop for each frame
layer_features = layers[CONTENT_LAYER]
"""
# Add an Op to initialize global variables.
init_op = tf.global_variables_initializer()

# Launch the graph in a session.
with tf.Session() as sess:

    # Run the Op that initializes global variables.
    sess.run(init_op)
"""
    # extract features from all frames (in batch) and compute content loos to each
init_target = []
h = 0

for b in range(BATCH_SIZE):
    content_frames_features = layer_features[b, :, :, :]

    target_frames_features = layer_features[BATCH_SIZE + b + 1:, :, :, :]

    c_loss.append(content_loss(CONTENT_WEIGHT, content_frames_features, target_frames_features))   # take features from content layer for generated images and content images and then compute content loss


init_target = np.random.uniform(0, 255, size=(BATCH_SIZE, height, width, 3))  # generate random input images to make style and content transfer on it

C_LOSS = tf.stack(c_loss)   # this is the tensor with content losses with frames around one batch

"""
    # z = sess.run(C_LOSS, feed_dict={target_ph: init_target[0:BATCH_SIZE], content_ph: frames[0:BATCH_SIZE]})   # vizualizaton
    # print(z)
"""

# Style loss

s_loss = []

def gram_matrix(features, normalize=True):
    shape = features.get_shape()
    num_channels = int(shape[2])
    matrix = tf.reshape(features, shape=[-1, num_channels])
    gram = tf.matmul(tf.transpose(matrix), matrix)
    if normalize == True:
        gram = tf.divide(gram, tf.to_float(tf.size(features)))
    return gram

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = height * width
    return tf.reduce_sum(tf.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

"""
# Add an Op to initialize global variables.
init_op = tf.global_variables_initializer()

# Launch the graph in a session.
with tf.Session() as sess:

    # Run the Op that initializes global variables.
    sess.run(init_op)
"""

for b in range(BATCH_SIZE):
    temp_loss = 0
    for layer_name in STYLE_LAYERS:
        layer_features = layers[layer_name]
        style_features = layer_features[BATCH_SIZE, :, :, :]
        combination_features = layer_features[BATCH_SIZE + b + 1, :, :, :]
        sl = style_loss(style_features, combination_features)
        temp_loss += (STYLE_WEIGHT/ len(STYLE_LAYERS)) * sl
    s_loss.append(temp_loss)


S_LOSS = tf.stack(s_loss)  # this is the tensor with content losses with frames around one batch


"""
    z = sess.run(S_LOSS, feed_dict={target_ph: init_target[0:BATCH_SIZE], content_ph: frames[0:BATCH_SIZE]})   # vizualizaton
    y = sess.run(C_LOSS, feed_dict={target_ph: init_target[0:BATCH_SIZE], content_ph: frames[0:BATCH_SIZE]})   # vizualizaton
    print(z)
    print(y)
"""

# Total variation loss

def total_variation_loss(img, tv_weight):
    L = tv_weight * tf.reduce_sum(tf.image.total_variation(img))
    return L
"""
# Add an Op to initialize global variables.
init_op = tf.global_variables_initializer()

# Launch the graph in a session.
with tf.Session() as sess:

    # Run the Op that initializes global variables.
    sess.run(init_op)
"""
tv_loss = []
for b in range(BATCH_SIZE):
    tv_loss.append(total_variation_loss(target_ph[b,:,:,:], TV_WEIGHT))

TV_LOSS = tf.stack(tv_loss)  # this is the tensor with content losses with frames around one batch

"""
    z = sess.run(TV_LOSS, feed_dict={target_ph: init_target[0:BATCH_SIZE], content_ph: frames[0:BATCH_SIZE]})   # vizualizaton
    print(z)
"""

# add all losses into one
LOSS = C_LOSS + S_LOSS + TV_LOSS
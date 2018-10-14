from __future__ import print_function
import cv2
import time
from PIL import Image
import numpy as np
import os
from keras import backend
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image

from scipy import optimize
from scipy.misc import imsave

import tensorflow as tf
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

height = 200
width = 200

def get_frames(video_path, height, width):
    capture = cv2.VideoCapture(video_path)
    success, image = capture.read()
    images = []
    count = 0
    while success:
        cv2.imwrite("../frames/frame%d.jpg" % count, image)
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

# make frames from content video
frames = np.array(get_frames('../video/Barry.mp4', height, width))
frames_count = 0
temp_frames = []
for frame in frames:
    temp_frames.append(preprocess(frame, True))
    frames_count += 1
#content_images = tf.Variable(np.asarray(temp_frames), dtype=tf.float32)

# have to make placehodler with no knowing one dimension of frames to concet inputs
content_image = tf.placeholder(tf.float32, shape=(None, height, width, 3))

# load style image
style_path = '../style/krzyk.jpg'
style = preprocess(load_img(style_path), True)
# style = np.expand_dims(style, axis=0)  # make another axis for frames index and to concat with other variables
style_image = tf.Variable(style)

# make placeholder for our target new frames
combination_image = tf.placeholder(tf.float32, shape=(None, height, width, 3))

# inputs for CNN
input_tensor = tf.concat([content_image,
                          style_image,
                          combination_image], axis=0)

model = VGG16(input_tensor=input_tensor, weights='imagenet',
              include_top=False)

layers = dict([(layer.name, layer.output) for layer in model.layers])


content_weight = 0.025
style_weight = 5.0
total_variation_weight = 1.0

loss = backend.variable(0.)

def content_loss(content, combination):
    return backend.sum(backend.square(combination - content))

layer_features = layers['block2_conv2']
content_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]

loss += content_weight * content_loss(content_image_features,
                                      combination_features)

def gram_matrix(x):
    features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
    gram = backend.dot(features, backend.transpose(features))
    return gram

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = height * width
    return backend.sum(backend.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

feature_layers = ['block1_conv2', 'block2_conv2',
                  'block3_conv3', 'block4_conv3',
                  'block5_conv3']

for layer_name in feature_layers:
    layer_features = layers[layer_name]
    style_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_features, combination_features)
    loss += (style_weight / len(feature_layers)) * sl

def total_variation_loss(x):
    a = backend.square(x[:, :height-1, :width-1, :] - x[:, 1:, :width-1, :])
    b = backend.square(x[:, :height-1, :width-1, :] - x[:, :height-1, 1:, :])
    return backend.sum(backend.pow(a + b, 1.25))

# TODO : Optical flow
# We have to take the optical flow into account and initialize the optimiaztion
# for the frame i + 1 with the previous stylized frame warped using the optical
# flow field that was estimeted between image x[i] and x[i+1]

"""
OpenCV provides algorithm to find the dense optical flow. 
It computes the optical flow for all the points in the frame. 
It is based on Gunner Farneback’s algorithm which 
is explained in “Two-Frame Motion Estimation Based on Polynomial Expansion” by Gunner Farneback in 2003.
"""

def Flow_forward(prev,next):
    return cv2.calcOpticalFlowFarneback(prev,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

def Flow_backward(prev,next):
    return cv2.calcOpticalFlowFarneback(next,prev, None, 0.5, 3, 15, 3, 5, 1.2, 0)

def warp(img,flow):
    h, w = flow.shape[:2]
    flow = - flow
    flow[:,:, 0] += np.arange(w)
    flow[:,:, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr
'''
def W(prev,next):
    return Flow_forward((prev,next) + Flow_backward(prev,next))


def temporal_loss(combination):
    W =
'''


loss += total_variation_weight * total_variation_loss(combination_image)

grads = backend.gradients(loss, combination_image)
print(grads)
outputs = [loss]
if type(grads) in {list, tuple}:
    outputs += grads
else:
    outputs.append(grads)

out_frames = []
x = np.random.uniform(0, 255, (1, height, width, 3)) - 128.


#TEST

'''
f_prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
f_next = cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY)
flow = Flow_forward(f_prev, f_next)
# the next initalization will be x
z = warp(f_prev, flow)

hsv = draw_hsv(flow)
cv2.imwrite("prev.jpg",frames[0])
cv2.imwrite("next.jpg", frames[1])
cv2.imwrite("hsv.jpg", hsv)
cv2.imwrite("wraped.jpg", z)
'''
#TEST

def eval_loss_and_grads(x):
    x = x.reshape((1, height, width, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

for i in range(10):

    f_outputs = backend.function([combination_image], outputs, feed_dict = {content_image: np.asarray(temp_frames[50])})
    evaluator = Evaluator()
    iterations = 7

    for i in range(iterations):
        print('Start of iteration', i)
        start_time = time.time()

        x, min_val, info = optimize.fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                         fprime=evaluator.grads, maxfun=20)

        print('Current loss value:', min_val)
        end_time = time.time()
        print('Iteration %d completed in %ds' % (i, end_time - start_time))

    prev = x
    x = x.reshape((height, width, 3))
    x = x[:, :, ::-1]
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = np.clip(x, 0, 255).astype('uint8')

    im = Image.fromarray(x)

    im.show()
    out_frames.append(x)

    # have to initialize the optimalization for the wraped frame

    if(i + 1 < frames_count):
        f_prev = cv2.cvtColor(frames[50],cv2.COLOR_BGR2GRAY)
        f_next = cv2.cvtColor(frames[50], cv2.COLOR_BGR2GRAY)
        flow = Flow_forward(f_prev, f_next)
        # the next initalization will be x
        x = warp(x, flow)
        cv2.imwrite("x_wraped.jpg", x)
    else:
        x = prev

video_name = 'test_same_frame_it_7_warped_krzyk.avi'
image_folder = 'video'

fourcc = cv2.VideoWriter_fourcc(*'MPEG')
video = cv2.VideoWriter(video_name,fourcc, 20.0, (width,height))

for ima in out_frames:
    video.write(cv2.cvtColor(ima,cv2.COLOR_RGB2BGR))

cv2.destroyAllWindows()
video.release()


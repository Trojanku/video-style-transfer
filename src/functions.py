import cv2
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras import backend
import tensorflow as tf
import numpy as np


def get_frames(video_path, height, width):
    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    success, image = capture.read()
    images = []
    count = 0
    while success:
        cv2.imwrite("../frames_org/frame%d.jpg" % count, image)
        images.append(cv2.resize(image, (height, width)))
        success, image = capture.read()
        count += 1
    return images, fps


def load_img(img_path,height,width):
    img = image.load_img(img_path, target_size=(height, width))
    return img


def preprocess(img, exp,height,width):
    x = image.img_to_array(img)
    if exp:
        x = np.expand_dims(x, axis=0)
    # new dims for number in model
    x = x.reshape((1, height, width, 3))
    x = preprocess_input(x)
    return x

def deprocess(x,height,width):
    x = x.reshape(height,width, 3)
    x = x[:, :, ::-1]
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def content_loss(content, combination):
    return backend.sum(backend.square(combination - content))


def gram_matrix(x):
    features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
    gram = backend.dot(features, backend.transpose(features))
    return gram


def style_loss(style, combination,height,width):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = height * width
    return backend.sum(backend.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


def total_variation_loss(x,height,width):
    a = backend.square(x[:, :height - 1, :width - 1, :] - x[:, 1:, :width - 1, :])
    b = backend.square(x[:, :height - 1, :width - 1, :] - x[:, :height - 1, 1:, :])
    return backend.sum(backend.pow(a + b, 1.25))

def neighbour_loss(prev, combination):
    return backend.sum(backend.square(prev - combination))


def optical_flow(prev,next):
    return cv2.calcOpticalFlowFarneback(prev,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)


def warp(img,flow):
    h, w = flow.shape[:2]
    flow = - flow
    flow[:,:, 0] += np.arange(w)
    flow[:,:, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

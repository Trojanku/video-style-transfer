import cv2
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras import backend
import numpy as np

height = 500
width = 500

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
    x = x.reshape((1, height, width, 3))
    x = preprocess_input(x)
    return x


def content_loss(content, combination):
    return backend.sum(backend.square(combination - content))


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


def total_variation_loss(x):
    a = backend.square(x[:, :height-1, :width-1, :] - x[:, 1:, :width-1, :])
    b = backend.square(x[:, :height-1, :width-1, :] - x[:, :height-1, 1:, :])
    return backend.sum(backend.pow(a + b, 1.25))


def neighbour_loss(prev, combination):
    return backend.sum(backend.square(prev - combination))

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
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

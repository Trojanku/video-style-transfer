import cv2
import numpy as np
import keras
import tensorflow as tf
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model


def get_frames(video_path):

    capture = cv2.VideoCapture(video_path)
    success,image = capture.read()
    images = []
    count = 0
    while success:
        cv2.imwrite("../frames/frame%d.jpg" % count,image)
        images.append(image)
        success, image = capture.read()
        count += 1
    return images

def load_img(img_path):
    img = image.load_img(img_path, target_size=(height, width))
    return preprocess(img)


def preprocess(img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

height = 255
width = 255

# make frames from content video
frames = np.array(get_frames('../video/Barry.mp4'))
frames_count = 0
for frame in frames:
    frames[frames_count] = preprocess(frame)
    frames_count = + 1
content_frames = tf.Variable(frames)

# load style image
style_path = '../style/gog.jpg'
style = load_img(style_path)
style_image = tf.Variable(style)

#make placeholder for our target array of new frames
target_frames = tf.placeholder(tf.float32, shape=(frames_count, height, width, 3))

# define CNN model
model = VGG19(weights='imagenet', include_top=False)  # include_top = False - > we dont need any full conected layers

layers = dict([(layer.name, layer.output) for layer in model.layers])


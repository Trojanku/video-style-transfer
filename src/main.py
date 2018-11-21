from __future__ import print_function
import cv2
import time
from PIL import Image
import numpy as np
from keras import backend
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from scipy import optimize

import tensorflow as tf
from tensorflow.python.client import device_lib

# --------------- Import from files ------------------
from Evaluator import Evaluator
from functions import *
# ----------------------------------------------------

# ---------------------- PyQt ------------------------
from PyQt5 import QtCore
import sys
from PyQt5.QtGui import QPixmap
from PIL.ImageQt import ImageQt
# ----------------------------------------------------

class style_transfer(QtCore.QObject):

    signal = QtCore.pyqtSignal(Image.Image)

    def __init__(self, style_path, content_path, iter, outlb,
                 content_weight, style_weight,neighbour_weight
                 ,width,height,style_layers,content_layers):

        QtCore.QObject.__init__(self)

        self.style_path = style_path
        self.content_path = content_path
        self.iterations = iter
        self.outlb = outlb
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.neighbour_weight = neighbour_weight
        self.width = width
        self.height = height
        self.style_layers = style_layers
        self.contet_layers = content_layers
        self.fps = 20

    def run(self):

        print(device_lib.list_local_devices())

        # make frames from content video
        frames, self.fps = get_frames(self.content_path, self.height, self.width)
        print('Content wideo fps:', self.fps)

        frames = np.asarray(frames)
        frames_count = 0
        temp_frames = []
        for frame in frames:
            temp_frames.append(preprocess(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), True,self.width,self.height))
            frames_count += 1
        print('wideo frames: ', frames_count)

        # have to make placehodler with no knowing one dimension of frames to concet inputs
        content_image = tf.placeholder(tf.float32, shape=(None, self.height, self.width, 3))

        # load style image
        style_path = self.style_path
        style = preprocess(load_img(style_path,self.width,self.height), True, self.width,self.height)
        style_image = tf.Variable(style)

        # make placeholder for our target new frames
        combination_image = tf.placeholder(tf.float32, shape=(None, self.height, self.width, 3))

        # make placehholder for previous generated frame to develop video loss
        previous_combination = tf.placeholder(tf.float32, shape=(1, self.height, self.width, 3))

        # inputs for CNN
        input_tensor = tf.concat([content_image,
                                  style_image,
                                  combination_image], axis=0)

        model = VGG16(input_tensor=input_tensor, weights='imagenet',
                      include_top=False)

        layers = dict([(layer.name, layer.output) for layer in model.layers])


        loss = backend.variable(0.)


        #layer_features = layers['block3_conv2']
        layer_features = layers[self.contet_layers[0]]

        content_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]

        # computer content loss
        loss += self.content_weight * content_loss(content_image_features,
                                              combination_features)

        #feature_layers = ['block1_conv2', 'block2_conv2',
         #                 'block3_conv3', 'block4_conv3',
         #                 'block5_conv3']
        #layer_features = self.style_layers

        for layer_name in self.style_layers:
            layer_features = layers[layer_name]
            style_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            sl = style_loss(style_features, combination_features,self.width,self.height)
            loss += (self.style_weight / len(self.style_layers)) * sl


        # add neigbour losses
        loss += self.neighbour_weight * neighbour_loss(previous_combination, combination_image)

        total_variation_weight = 1.
        loss += total_variation_weight * total_variation_loss(combination_image,self.width,self.height)


        grads = backend.gradients(loss, combination_image)
        outputs = [loss]
        if type(grads) in {list, tuple}:
            outputs += grads
        else:
            outputs.append(grads)

        out_frames = []
        x = np.random.uniform(0, 255, (1, self.height, self.width, 3)) - 128.
        prev = x

        for i in range(frames_count):
            f_outputs = backend.function([combination_image, previous_combination], outputs, feed_dict = {
                                                                                    content_image: temp_frames[i]})
            evaluator = Evaluator(f_outputs,self.height,self.width)
            if i < 1:
                iterations = self.iterations
            else:
                iterations = 8

            print('Current frame: ', i)
            for j in range(iterations):
                print('Start of iteration', j)
                start_time = time.time()

                x, min_val, info = optimize.fmin_l_bfgs_b(evaluator.loss, x.flatten(), args=(prev, ),
                                                 fprime=evaluator.grads, maxfun=20)

                print('Current loss value:', min_val)
                end_time = time.time()
                print('Iteration %d completed in %ds' % (j, end_time - start_time))


            x = x.reshape((self.height, self.width, 3))

            # this part undo preprocess ( add mean values of imagenet pixels RGB )
            x = x[:, :, ::-1]
            x[:, :, 0] += 103.939
            x[:, :, 1] += 116.779
            x[:, :, 2] += 123.68
            x = np.clip(x, 0, 255).astype('uint8')


            im = Image.fromarray(x)
            im.save('actual_frame.jpg')

            im = ImageQt(im)
            pix = QPixmap.fromImage(im)
            self.outlb.setPixmap(pix)
            out_frames.append(x)


            # have to initialize the optimalization for the wraped frame

            if(i + 1 < frames_count):
                f_prev = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                f_next = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
                flow = Flow_forward(f_prev, f_next)
                # the next initalization will be x

                x = preprocess(warp(x, flow),True,self.width,self.height)
                cv2.imwrite("x_wraped.jpg", x)
            else:
                x = preprocess(x, True,self.width,self.height)

            prev = x

        video_name = '../generated/out.avi'

        fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        video = cv2.VideoWriter(video_name,fourcc, int(self.fps), (self.width,self.height))

        for ima in out_frames:
            video.write(cv2.cvtColor(ima,cv2.COLOR_RGB2BGR))

        cv2.destroyAllWindows()
        video.release()
        backend.clear_session()



from __future__ import print_function
import time
from PIL import Image
from keras.applications.vgg16 import VGG16
from scipy import optimize
import tensorflow as tf
from tensorflow.python.client import device_lib
from Evaluator import Evaluator
from functions import *
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap
from PIL.ImageQt import ImageQt


class StyleTransfer(QtCore.QObject):
    signal = QtCore.pyqtSignal(Image.Image)

    def __init__(self):

        QtCore.QObject.__init__(self)

        self.style_layers = []
        self.contet_layers = []

    def set_values(self, style_path, content_path, iter, outlb, framelb, iterlb,
                   content_weight, style_weight, neighbour_weight
                   , width, height, style_layers, content_layers):

        self.style_path = style_path
        self.content_path = content_path
        self.iterations = iter
        self.outlb = outlb
        self.framelb = framelb
        self.iterlb = iterlb
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.neighbour_weight = neighbour_weight
        self.width = width
        self.height = height
        self.style_layers = style_layers
        self.contet_layers = content_layers
        self.fps = 20
        self.stop = False

    def end(self):
        self.stop = True

    def run(self):

        backend.clear_session()
        tf.reset_default_graph()
        cv2.destroyAllWindows()

        self.stop = False
        print(device_lib.list_local_devices())

        # make frames from content video
        frames, self.fps = get_frames(self.content_path, self.height, self.width)
        print('Content wideo fps:', self.fps)

        frames = np.asarray(frames)
        frames_count = 0
        temp_frames = []
        for frame in frames:
            temp_frames.append(preprocess(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), True, self.height, self.width))
            frames_count += 1
        print('wideo frames: ', frames_count)

        # have to make placehodler with no knowing one dimension of frames to concet inputs
        content_image = tf.placeholder(tf.float32, shape=(None, self.height, self.width, 3))

        # load style image
        style_path = self.style_path
        style = preprocess(load_img(style_path, self.height, self.width), True, self.height, self.width)
        style_image = tf.Variable(style)

        # make placeholder for our target new frames
        combination_image = tf.placeholder(tf.float32, shape=(None, self.height, self.width, 3))

        # make placehholder for previous generated frame to develop video loss
        previous_combination = tf.placeholder(tf.float32, shape=(1, self.height, self.width, 3))

        # inputs for CNN
        input_tensor = tf.concat([content_image,
                                  style_image,
                                  combination_image], axis=0)

        # define CNN model
        model = VGG16(input_tensor=input_tensor, weights='imagenet',
                      pooling=max, include_top=False)

        layers = dict([(layer.name, layer.output) for layer in model.layers])

        loss = backend.variable(0.)

        # add content loss
        layer_features = layers[self.contet_layers[0]]
        content_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]
        loss += self.content_weight * content_loss(content_image_features, combination_features)

        # add style loss
        for layer_name in self.style_layers:
            layer_features = layers[layer_name]
            style_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            loss += (self.style_weight / len(self.style_layers)) * style_loss(style_features, combination_features, self.height, self.width)

        # add neigbour losses
        loss += self.neighbour_weight * neighbour_loss(previous_combination, combination_image)

        # add total variation regularizer
        loss += total_variation_loss(combination_image, self.height, self.width)

        # compute gradients
        grads = backend.gradients(loss, combination_image)

        # create variable outputs to store loss and gradients
        outputs = [loss]
        outputs += grads

        out_frames = []
        x = np.random.uniform(0, 255, (1, self.height, self.width, 3)) - 128.
        prev = x.copy()
        z = x.copy()

        evaluator = Evaluator(self.height, self.width)

        for i in range(frames_count):

            # compute tensorflow graph to get loss value and gradients
            f_outputs = backend.function([combination_image, previous_combination], outputs, feed_dict={
                content_image: temp_frames[i]})

            evaluator.set_data(f_outputs)
            iterations = self.iterations

            print('Current frame: ', i)
            self.framelb.setText(str(i + 1))

            for j in range(iterations):

                if not self.stop:
                    print('Start of iteration', j)
                    self.iterlb.setText(str(j + 1))
                    start_time = time.time()

                    x, min_val, info = optimize.fmin_l_bfgs_b(evaluator.loss, x.flatten(), args=(prev.flatten(),),
                                                              fprime=evaluator.grads, maxfun=20)

                    print('Current loss value:', min_val)
                    end_time = time.time()
                    print('Iteration %d completed in %ds' % (j, end_time - start_time))

                    z = x.copy()
                    z = deprocess(z, self.height, self.width)
                    im = Image.fromarray(z)

                    # show actual generated image on gui
                    im = ImageQt(im)
                    pix = QPixmap.fromImage(im)
                    self.outlb.setPixmap(pix)

                else:
                    break

            out_frames.append(z)

            # have to initialize the optimalization for the wraped frame
            if (i + 1 < frames_count):
                f_prev = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                f_next = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
                flow = optical_flow(f_prev, f_next)
                # the next initalization will be x

                x = preprocess(warp(z, flow), True, self.height, self.width)
            else:
                x = preprocess(z, True, self.height, self.width)

            prev = x.copy()

        video_name = '../generated/out.avi'

        fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        video = cv2.VideoWriter(video_name, fourcc, int(self.fps), (self.height, self.width))

        for ima in out_frames:
            video.write(cv2.cvtColor(ima, cv2.COLOR_RGB2BGR))

        del evaluator
        cv2.destroyAllWindows()
        video.release()
        backend.clear_session()
        tf.reset_default_graph()

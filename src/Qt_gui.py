from PyQt5 import uic, QtWidgets, QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog, QLabel, QMainWindow
from PyQt5.QtMultimediaWidgets import  QVideoWidget
from PyQt5 import QtCore
from PIL import Image
from PIL.ImageQt import ImageQt
from main import style_transfer
import threading
import time
import numpy as np
import cv2
import sys

class App(QMainWindow):

    def __init__(self):
        super(App, self).__init__()
        uic.loadUi('../desing_main.ui', self)
        self.show()


        self.stylebt.clicked.connect(self.load_style)
        self.contentbt.clicked.connect(self.load_content)
        self.start.clicked.connect(self.process)

        self.style_layers = []
        self.content_layer = []

    def load_style(self):
        self.style_path = QFileDialog.getOpenFileName(self, 'Open file', '~/', "Image files (*.jpg *.gif)")
        self.style_image = QPixmap(self.style_path[0])
        self.stylelb.setPixmap(QPixmap( self.style_image))

    def load_content(self):
        self.content_path = QFileDialog.getOpenFileName(self, 'Open file', '~/')

        self.content_video = cv2.VideoCapture(self.content_path[0])
        success, frame = self.content_video.read()
        if(success):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.destroyAllWindows()
            im = Image.fromarray(frame)

            im = ImageQt(im.resize((300,300)))
            pix = QPixmap.fromImage(im)
            self.contentlb.setPixmap(pix)

    def add(self,image):
        im = ImageQt(image)
        pix = QPixmap.fromImage(im)
        self.outlb.setPixmap(pix)

    def get_style_layers(self):
        for item in self.style_layers_list.selectedItems():
            self.style_layers.append(item.text())
        print(self.style_layers)

    def get_content_layer(self):
        for item in self.content_layer_list.selectedItems():
            self.content_layer.append(item.text())
        print(self.content_layer)

    def process(self):

        iterations = self.iterationsSB.value()
        content_weight = self.content_weightSB.value()
        style_weight = self.style_weightSB.value()
        neighbour_weight = self.wideo_weightSB.value()
        width = self.widthSB.value()
        height = self.heightSB.value()

        self.get_content_layer()
        self.get_style_layers()

        transfer = style_transfer(self.style_path[0], self.content_path[0], iterations,
                                  self.outlb, content_weight, style_weight,
                                  neighbour_weight, width, height, self.style_layers,
                                  self.content_layer)

        t = threading.Thread(target=transfer.run)

        t.start()

        del t
        del transfer


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = App()
    sys.exit(app.exec_())
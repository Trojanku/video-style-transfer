from PyQt5 import uic, QtWidgets
from PyQt5.QtGui import QPixmap,QImage
from PyQt5.QtWidgets import QFileDialog, QMainWindow
from PyQt5.QtCore import Qt
from PIL import Image
from PIL.ImageQt import ImageQt
from transfer import StyleTransfer
import threading
import cv2
import sys

class App(QMainWindow):

    def __init__(self):
        super(App, self).__init__()
        uic.loadUi('../desing_main.ui', self)
        self.show()


        self.stylebt.clicked.connect(self.load_style)
        self.contentbt.clicked.connect(self.load_content)
        self.startb.clicked.connect(self.process)
        self.stopb.clicked.connect(self.stop)
        self.label_klatka_max.text = '0'
        self.style_layers = []
        self.content_layer = []

        self.transfer = StyleTransfer()

    def count_frames(self,video):
        total = 0
        first = 0
        while True:
            (grabbed, frame) = video.read()
            if total == 0:
                first = frame
            if not grabbed:
                break
            total += 1
        return total, first

    def load_style(self):
        self.style_path = QFileDialog.getOpenFileName(self, 'Open file', '~/', "Image files (*.jpg *.gif *.png *.JPG)")
        self.style_image = QPixmap(self.style_path[0])
        self.stylelb.setPixmap(QPixmap( self.style_image))

    def load_content(self):
        self.content_path = QFileDialog.getOpenFileName(self, 'Open file', '~/',
                                                        "Video/Image files (*.avi *.mp4 *.wave *.jpg *.gif *.png *.JPG)")

        content_video = cv2.VideoCapture(self.content_path[0])
        length, first = self.count_frames(content_video)
        self.label_klatka_max.setText(str(length))
        cv2.VideoCapture.release(content_video)

        frame = cv2.cvtColor(first, cv2.COLOR_RGB2BGR)

        im = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        p = im.scaled(200, 200, Qt.KeepAspectRatio)

        self.contentlb.setPixmap(QPixmap(p))

        cv2.destroyAllWindows()

    def add(self,image):
        im = ImageQt(image)
        pix = QPixmap.fromImage(im)
        self.outlb.setPixmap(pix)

    def get_style_layers(self):
        self.style_layers = []
        for item in self.style_layers_list.selectedItems():
            self.style_layers.append(item.text())
        print(self.style_layers)

    def get_content_layer(self):
        self.content_layers = []
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

        self.label_iteracja_max.setText(str(iterations))

        self.transfer.set_values(self.style_path[0], self.content_path[0], iterations,
                                  self.outlb, self.label_frame, self.label_iter,
                                  content_weight, style_weight,
                                  neighbour_weight, width, height, self.style_layers,
                                  self.content_layer)

        self.t = threading.Thread(target=self.transfer.run, daemon=True)
        self.t.start()
        del self.t

    def stop(self):
        if self.transfer:
            self.transfer.end()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = App()
    sys.exit(app.exec_())
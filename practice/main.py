import math
import matplotlib.pyplot as plt
import matplotlib.image as mping
import cv2
import numpy as np
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QBoxLayout
from PyQt5 import QtCore
from PyQt5.QtCore import QObject

class Test(QObject):
    def __init__(self):
        super().__init__()
        self.cnt = 0
        self.stop_flag = False

    def img_read(self):
        while True:
            self.cnt +=1
            if self.cnt%2==0:
                self.img = cv2.imread('img/0_0.jpg')
                print(self.cnt,'img 0 read')

            else:
                self.img = cv2.imread('img/0_90.jpg')
                print(self.cnt,'img 90 read')

            loop = QtCore.QEventLoop()
            QtCore.QTimer.singleShot(1000, loop.quit)  # 1000 ms
            loop.exec_()

            if self.stop_flag:
                self.stop_flag = False
                break


    def stop_test(self):
        self.stop_flag = True

class Test2(QObject):
    def __init__(self):
        super().__init__()
        self.cnt2 = 0
        # self.stop_flag = False

    def start_test2(self):
        while True:
            self.cnt2 += 1
            print('test2 = ', self.cnt2)

            loop = QtCore.QEventLoop()
            QtCore.QTimer.singleShot(1000, loop.quit) #1000 ms
            loop.exec_()

    def show_img(self):
        while True:
            imshow(self.img)
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.thread = QtCore.QThread()
        self.thread.start()
        self.test = Test()
        self.test.moveToThread(self.thread)

        self.thread2 = QtCore.QThread()
        self.thread2.start()
        self.test2 = Test2()
        self.test2.moveToThread(self.thread2)

        self.pb1 = QPushButton("start1")
        self.pb2 = QPushButton("stop")
        self.pb3 = QPushButton("start2")
        # self.test.start_test()

        form_lbx = QBoxLayout(QBoxLayout.TopToBottom, self)
        self.setLayout(form_lbx)

        form_lbx.addWidget(self.pb1)
        form_lbx.addWidget(self.pb2)
        form_lbx.addWidget(self.pb3)

        self.pb1.clicked.connect(self.test.img_read)
        self.pb2.clicked.connect(self.test.stop_test)
        self.pb3.clicked.connect(self.test2.start_test2)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    form = MainWindow()
    form.show()
    app.exec_()
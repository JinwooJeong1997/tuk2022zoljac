# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'tool_layout_ui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication,QDialog
import sys
from PyQt5.QtCore import QObject, QThread
import detect_circle as d
import server as s
import camera as c
import UI as u

# thread1 : detect_circle.py
# thread2 : UI.py
# thread3 : camera.py
# thread4 : server.py

class Main(QObject):

    def __init__(self, parent=None):
        super(self.__class__, self).__init__(parent)

        self.thread = QThread()
        self.thread.start()
        self.calc = d.Circle()
        self.calc.moveToThread(self.thread)

        self.thread2 = QThread()
        self.thread2.start()
        self.gui = u.Ui_MainWindow()
        self.gui.moveToThread(self.thread2)

        self.thread3 = QThread()
        self.thread3.start()
        self.camera = c.Camera()
        self.camera.moveToThread(self.thread3)

        self.thread4 = QThread()
        self.thread4.start()
        self.server = s.Server()
        self.server.moveToThread(self.thread4)

        self.gui.setupUi(MainWindow)
        self._connectSignals()

    def _connectSignals(self):
        self.gui.start.clicked.connect(self.calc.OnOpenDocument)
        self.calc.sig.connect(self.gui.show_img)
        self.gui.camera_info.clicked.connect(self.camera.camera)
        self.gui.server_info.clicked.connect(self.server.server)
        self.gui.HB_Val.currentIndexChanged.connect(self.gui.change_HB)

    def forceWorkerReset(self):
        if self.worker_thread.isRunning():
            self.worker_thread.terminate()
            self.worker_thread.wait()
            self.worker_thread.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    example=Main(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())


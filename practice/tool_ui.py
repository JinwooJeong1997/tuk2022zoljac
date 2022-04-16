# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'tool.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QBoxLayout
import sys
from PyQt5.QtCore import QObject,pyqtSignal, pyqtSlot, QThread
import time


class Test(QObject):
    sig=pyqtSignal(str)

    def __init__(self,parent=None):
        super(self.__class__, self).__init__(parent)


    @pyqtSlot()
    def img_read(self):
        self.cnt = 0
        while True:
            self.cnt +=1
            if self.cnt%2==0:
                a='img/0_0.jpg'
                print(self.cnt,'img 0 read')

            else:
                a='img/90_0.jpg'
                print(self.cnt,'img 90 read')

            self.sig.emit(a)
            time.sleep(1)


class Example(QObject):

    def __init__(self, parent=None):
        super(self.__class__, self).__init__(parent)

        self.gui = Ui_MainWindow()

        self.thread = QThread()
        self.thread.start()
        self.test = Test()
        self.test.moveToThread(self.thread)

        self.thread2 = QThread()
        self.thread2.start()
        #self.gui.moveToThread(self.thread2)



        self.gui.setupUi(MainWindow)

        self._connectSignals()


    def _connectSignals(self):
        self.gui.button_start.clicked.connect(self.test.img_read)
        self.test.sig.connect(self.gui.show_img)



    def forceWorkerReset(self):
        if self.worker_thread.isRunning():
            self.worker_thread.terminate()
            self.worker_thread.wait()
            self.worker_thread.start()

class Ui_MainWindow(QWidget):

    def setupUi(self,MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(799, 807)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.title1 = QtWidgets.QLabel(self.centralwidget)
        self.title1.setGeometry(QtCore.QRect(40, 0, 231, 61))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(22)
        font.setBold(True)
        font.setWeight(75)
        self.title1.setFont(font)
        self.title1.setObjectName("title1")
        self.top_line = QtWidgets.QFrame(self.centralwidget)
        self.top_line.setGeometry(QtCore.QRect(0, 130, 801, 16))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(12)
        self.top_line.setFont(font)
        self.top_line.setLineWidth(2)
        self.top_line.setFrameShape(QtWidgets.QFrame.HLine)
        self.top_line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.top_line.setObjectName("top_line")
        self.title2 = QtWidgets.QLabel(self.centralwidget)
        self.title2.setGeometry(QtCore.QRect(90, 60, 131, 41))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(22)
        font.setBold(True)
        font.setWeight(75)
        self.title2.setFont(font)
        self.title2.setObjectName("title2")
        self.company_logo = QtWidgets.QLabel(self.centralwidget)
        self.company_logo.setGeometry(QtCore.QRect(300, 0, 151, 71))
        self.company_logo.setObjectName("company_logo")
        self.company_name = QtWidgets.QLabel(self.centralwidget)
        self.company_name.setGeometry(QtCore.QRect(280, 80, 211, 31))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.company_name.setFont(font)
        self.company_name.setObjectName("company_name")
        self.server = QtWidgets.QLabel(self.centralwidget)
        self.server.setGeometry(QtCore.QRect(530, 20, 61, 31))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.server.setFont(font)
        self.server.setObjectName("server")
        self.camera = QtWidgets.QLabel(self.centralwidget)
        self.camera.setGeometry(QtCore.QRect(530, 70, 71, 31))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.camera.setFont(font)
        self.camera.setObjectName("camera")
        self.main_image = QtWidgets.QLabel(self.centralwidget)
        self.main_image.setGeometry(QtCore.QRect(25, 181, 421, 301))
        self.main_image.setObjectName("main_image")
        self.server_val = QtWidgets.QLabel(self.centralwidget)
        self.server_val.setGeometry(QtCore.QRect(625, 22, 101, 31))
        self.server_val.setObjectName("server_val")
        self.camera_val = QtWidgets.QLabel(self.centralwidget)
        self.camera_val.setGeometry(QtCore.QRect(620, 70, 101, 31))
        self.camera_val.setObjectName("camera_val")
        self.errors = QtWidgets.QGroupBox(self.centralwidget)
        self.errors.setGeometry(QtCore.QRect(460, 180, 311, 301))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.errors.setFont(font)
        self.errors.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.errors.setFlat(False)
        self.errors.setObjectName("errors")
        self.error_val = QtWidgets.QLabel(self.errors)
        self.error_val.setGeometry(QtCore.QRect(80, 40, 141, 51))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.error_val.setFont(font)
        self.error_val.setObjectName("error_val")
        self.tooler_error = QtWidgets.QLabel(self.errors)
        self.tooler_error.setGeometry(QtCore.QRect(50, 120, 131, 21))
        self.tooler_error.setObjectName("tooler_error")
        self.matter_error = QtWidgets.QLabel(self.errors)
        self.matter_error.setGeometry(QtCore.QRect(50, 170, 131, 21))
        self.matter_error.setObjectName("matter_error")
        self.tool_life_error = QtWidgets.QLabel(self.errors)
        self.tool_life_error.setGeometry(QtCore.QRect(50, 220, 131, 21))
        self.tool_life_error.setObjectName("tool_life_error")
        self.red_green_1 = QtWidgets.QLabel(self.errors)
        self.red_green_1.setGeometry(QtCore.QRect(200, 120, 81, 21))
        self.red_green_1.setObjectName("red_green_1")
        self.red_green_2 = QtWidgets.QLabel(self.errors)
        self.red_green_2.setGeometry(QtCore.QRect(200, 160, 81, 21))
        self.red_green_2.setObjectName("red_green_2")
        self.red_green_3 = QtWidgets.QLabel(self.errors)
        self.red_green_3.setGeometry(QtCore.QRect(200, 220, 81, 21))
        self.red_green_3.setObjectName("red_green_3")

        self.info = QtWidgets.QGroupBox(self.centralwidget)
        self.info.setGeometry(QtCore.QRect(460, 500, 301, 141))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.info.setFont(font)
        self.info.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.info.setFlat(False)
        self.info.setObjectName("info")
        self.round = QtWidgets.QLabel(self.info)
        self.round.setGeometry(QtCore.QRect(50, 50, 61, 21))
        self.round.setObjectName("round")
        self.ratate = QtWidgets.QLabel(self.info)
        self.ratate.setGeometry(QtCore.QRect(50, 90, 51, 21))
        self.ratate.setObjectName("ratate")
        self.round_val = QtWidgets.QLabel(self.info)
        self.round_val.setGeometry(QtCore.QRect(140, 50, 81, 21))
        self.round_val.setObjectName("round_val")
        self.rotate_val = QtWidgets.QLabel(self.info)
        self.rotate_val.setGeometry(QtCore.QRect(140, 80, 81, 21))
        self.rotate_val.setObjectName("rotate_val")
        self.button_start = QtWidgets.QPushButton('Start', self.centralwidget)
        self.button_start.setGeometry((QtCore.QRect(500,650,100,100)))
        self.Tool_life = QtWidgets.QGroupBox(self.centralwidget)
        self.Tool_life.setGeometry(QtCore.QRect(20, 500, 431, 271))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.Tool_life.setFont(font)
        self.Tool_life.setAlignment(QtCore.Qt.AlignCenter)
        self.Tool_life.setObjectName("Tool_life")
        self.tool_img = QtWidgets.QLabel(self.Tool_life)
        self.tool_img.setGeometry(QtCore.QRect(90, 20, 261, 221))
        self.tool_img.setText("")
        self.tool_img.setObjectName("tool_img")
        self.cut_num = QtWidgets.QLabel(self.Tool_life)
        self.cut_num.setGeometry(QtCore.QRect(180, 80, 91, 21))
        self.cut_num.setObjectName("cut_num")
        self.wear = QtWidgets.QLabel(self.Tool_life)
        self.wear.setGeometry(QtCore.QRect(190, 140, 51, 21))
        self.wear.setObjectName("wear")
        self.cut_num_val = QtWidgets.QLabel(self.Tool_life)
        self.cut_num_val.setGeometry(QtCore.QRect(190, 100, 51, 21))
        self.cut_num_val.setObjectName("cut_num_val")
        self.wear_val = QtWidgets.QLabel(self.Tool_life)
        self.wear_val.setGeometry(QtCore.QRect(190, 160, 41, 21))
        self.wear_val.setObjectName("wear_val")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 799, 21))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menu.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.title1.setText(_translate("MainWindow", "절단봉 불량 검출"))
        self.title2.setText(_translate("MainWindow", "프로그램"))
        self.company_name.setText(_translate("MainWindow", "COMPANY MOTORS"))
        self.server.setText(_translate("MainWindow", "server"))
        self.camera.setText(_translate("MainWindow", "camera"))
        self.server_val.setText(_translate("MainWindow", "//////////////"))
        self.camera_val.setText(_translate("MainWindow", "//////////////"))
        self.errors.setTitle(_translate("MainWindow", "불량검출"))
        self.error_val.setText(_translate("MainWindow", "  NG / GOOD"))
        self.tooler_error.setText(_translate("MainWindow", "절단봉 툴러 불량"))
        self.matter_error.setText(_translate("MainWindow", "절단봉 이물질"))
        self.tool_life_error.setText(_translate("MainWindow", "툴라이프 수명"))
        self.red_green_1.setText(_translate("MainWindow", "/////"))
        self.red_green_2.setText(_translate("MainWindow", "/////"))
        self.red_green_3.setText(_translate("MainWindow", "/////"))
        self.info.setTitle(_translate("MainWindow", "절단봉 정보"))
        self.round.setText(_translate("MainWindow", "- 둘레"))
        self.ratate.setText(_translate("MainWindow", "- 회전"))
        self.round_val.setText(_translate("MainWindow", "/////"))
        self.rotate_val.setText(_translate("MainWindow", "/////"))
        self.Tool_life.setTitle(_translate("MainWindow", "Tool-Life"))
        self.cut_num.setText(_translate("MainWindow", "절단 횟수"))
        self.wear.setText(_translate("MainWindow", "마모도"))
        self.cut_num_val.setText(_translate("MainWindow", "100 번"))
        self.wear_val.setText(_translate("MainWindow", "30 %"))
        self.menu.setTitle(_translate("MainWindow", "절단봉"))

    @pyqtSlot(str)
    def show_img(self, img):
        print('re')
        pixmap = QtGui.QPixmap(img)
        pixmap = pixmap.scaledToWidth(421)
        self.main_image.setPixmap(pixmap)

'''
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
'''
if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    example=Example(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

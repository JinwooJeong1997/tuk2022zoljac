# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'tool_layout_ui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

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
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(654, 616)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.tool_life_box = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.tool_life_box.setFont(font)
        self.tool_life_box.setAlignment(QtCore.Qt.AlignCenter)
        self.tool_life_box.setObjectName("tool_life_box")

        self.gridLayout_8 = QtWidgets.QGridLayout(self.tool_life_box)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.gridLayout_7 = QtWidgets.QGridLayout()
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.wear = QtWidgets.QLabel(self.tool_life_box)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.wear.setFont(font)
        self.wear.setAlignment(QtCore.Qt.AlignCenter)
        self.wear.setObjectName("wear")
        self.wear.setStyleSheet(stylesheet)
        self.gridLayout_7.addWidget(self.wear, 2, 0, 1, 1)
        self.wear_val = QtWidgets.QLabel(self.tool_life_box)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.wear_val.setFont(font)
        self.wear_val.setAlignment(QtCore.Qt.AlignCenter)
        self.wear_val.setObjectName("wear_val")
        self.gridLayout_7.addWidget(self.wear_val, 3, 0, 1, 1)
        self.cut_num = QtWidgets.QLabel(self.tool_life_box)
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.cut_num.setFont(font)
        self.cut_num.setAlignment(QtCore.Qt.AlignCenter)
        self.cut_num.setObjectName("cut_num")
        self.gridLayout_7.addWidget(self.cut_num, 0, 0, 1, 1)
        self.cut_num_val = QtWidgets.QLabel(self.tool_life_box)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.cut_num_val.setFont(font)
        self.cut_num_val.setAlignment(QtCore.Qt.AlignCenter)
        self.cut_num_val.setObjectName("cut_num_val")
        self.gridLayout_7.addWidget(self.cut_num_val, 1, 0, 1, 1)
        self.gridLayout_8.addLayout(self.gridLayout_7, 0, 0, 1, 1)
        self.gridLayout_5.addWidget(self.tool_life_box, 0, 0, 1, 1)
        self.info_box = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.info_box.setFont(font)
        self.info_box.setAlignment(QtCore.Qt.AlignCenter)
        self.info_box.setObjectName("info_box")
        self.gridLayout_10 = QtWidgets.QGridLayout(self.info_box)
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.gridLayout_9 = QtWidgets.QGridLayout()
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.rotate_val = QtWidgets.QLabel(self.info_box)
        self.rotate_val.setObjectName("rotate_val")
        self.gridLayout_9.addWidget(self.rotate_val, 1, 1, 1, 1)
        self.round_val = QtWidgets.QLabel(self.info_box)
        self.round_val.setObjectName("round_val")
        self.gridLayout_9.addWidget(self.round_val, 0, 1, 1, 1)
        self.rotate = QtWidgets.QLabel(self.info_box)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.rotate.setFont(font)
        self.rotate.setAlignment(QtCore.Qt.AlignCenter)
        self.rotate.setObjectName("rotate")
        self.gridLayout_9.addWidget(self.rotate, 1, 0, 1, 1)
        self.round = QtWidgets.QLabel(self.info_box)
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.round.setFont(font)
        self.round.setAlignment(QtCore.Qt.AlignCenter)
        self.round.setObjectName("round")
        self.gridLayout_9.addWidget(self.round, 0, 0, 1, 1)
        self.gridLayout_10.addLayout(self.gridLayout_9, 0, 0, 1, 1)
        self.button_start = QtWidgets.QPushButton(self.info_box)
        self.button_start.setObjectName("start")
        self.gridLayout_10.addWidget(self.button_start, 1, 0, 1, 1)
        self.stop = QtWidgets.QPushButton(self.info_box)
        self.stop.setObjectName("stop")
        self.gridLayout_10.addWidget(self.stop, 2, 0, 1, 1)
        self.gridLayout_5.addWidget(self.info_box, 0, 1, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout_5, 3, 0, 1, 1)
        self.gridlayout = QtWidgets.QGridLayout()
        self.gridlayout.setContentsMargins(-1, -1, 0, -1)
        self.gridlayout.setSpacing(6)
        self.gridlayout.setObjectName("gridlayout")
        self.error_box = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.error_box.setFont(font)
        self.error_box.setAlignment(QtCore.Qt.AlignCenter)
        self.error_box.setObjectName("error_box")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.error_box)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.gridLayout_11 = QtWidgets.QGridLayout()
        self.gridLayout_11.setObjectName("gridLayout_11")
        self.tool_life = QtWidgets.QLabel(self.error_box)
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.tool_life.setFont(font)
        self.tool_life.setObjectName("tool_life")
        self.gridLayout_11.addWidget(self.tool_life, 2, 0, 1, 1)
        self.tooler_error_val = QtWidgets.QLabel(self.error_box)
        self.tooler_error_val.setObjectName("tooler_error_val")
        self.gridLayout_11.addWidget(self.tooler_error_val, 0, 1, 1, 1)
        self.matter_val = QtWidgets.QLabel(self.error_box)
        self.matter_val.setObjectName("matter_val")
        self.gridLayout_11.addWidget(self.matter_val, 1, 1, 1, 1)
        self.tool_life_val = QtWidgets.QLabel(self.error_box)
        self.tool_life_val.setObjectName("tool_life_val")
        self.gridLayout_11.addWidget(self.tool_life_val, 2, 1, 1, 1)
        self.tooler_error = QtWidgets.QLabel(self.error_box)
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.tooler_error.setFont(font)
        self.tooler_error.setObjectName("tooler_error")
        self.gridLayout_11.addWidget(self.tooler_error, 0, 0, 1, 1)
        self.matter = QtWidgets.QLabel(self.error_box)
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.matter.setFont(font)
        self.matter.setObjectName("matter")
        self.gridLayout_11.addWidget(self.matter, 1, 0, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_11, 1, 0, 1, 1)
        self.ng_good = QtWidgets.QLabel(self.error_box)
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.ng_good.setFont(font)
        self.ng_good.setObjectName("ng_good")
        self.ng_good.setStyleSheet("background-color: #7FFFD4")
        self.gridLayout_4.addWidget(self.ng_good, 0, 0, 1, 1)
        self.gridLayout_6.addLayout(self.gridLayout_4, 0, 0, 1, 1)
        self.gridlayout.addWidget(self.error_box, 0, 1, 1, 1)
        self.img = QtWidgets.QLabel(self.centralwidget)
        self.img.setText("")
        self.img.setObjectName("img")
        self.gridlayout.addWidget(self.img, 0, 0, 1, 1)
        self.gridLayout_2.addLayout(self.gridlayout, 2, 0, 1, 1)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.name = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(26)
        font.setBold(True)
        font.setWeight(75)
        self.name.setFont(font)
        self.name.setObjectName("name")
        self.verticalLayout_9.addWidget(self.name)
        self.horizontalLayout_2.addLayout(self.verticalLayout_9)
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.logo = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(50)
        font.setBold(False)
        font.setWeight(40)
        self.logo.setFont(font)
        self.logo.setObjectName("logo")
        self.verticalLayout_8.addWidget(self.logo)
        self.co_name = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.co_name.setFont(font)
        self.co_name.setObjectName("co_name")
        self.verticalLayout_8.addWidget(self.co_name)
        self.horizontalLayout_2.addLayout(self.verticalLayout_8)
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setContentsMargins(-1, -1, 40, -1)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.server = QtWidgets.QLabel(self.centralwidget)
        self.server.setMaximumSize(QtCore.QSize(100, 50))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.server.setFont(font)
        self.server.setObjectName("server")
        self.gridLayout_3.addWidget(self.server, 0, 0, 1, 1)
        self.camera = QtWidgets.QLabel(self.centralwidget)
        self.camera.setMaximumSize(QtCore.QSize(100, 50))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.camera.setFont(font)
        self.camera.setObjectName("camera")
        self.gridLayout_3.addWidget(self.camera, 1, 0, 1, 1)
        self.camera_val = QtWidgets.QLabel(self.centralwidget)
        self.camera_val.setMaximumSize(QtCore.QSize(100, 50))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.camera_val.setFont(font)
        self.camera_val.setObjectName("camera_val")
        self.gridLayout_3.addWidget(self.camera_val, 1, 1, 1, 1)
        self.server_val = QtWidgets.QLabel(self.centralwidget)
        self.server_val.setMaximumSize(QtCore.QSize(100, 50))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.server_val.setFont(font)
        self.server_val.setObjectName("server_val")
        self.gridLayout_3.addWidget(self.server_val, 0, 1, 1, 1)
        self.horizontalLayout_2.addLayout(self.gridLayout_3)
        self.verticalLayout_5.addLayout(self.horizontalLayout_2)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setLineWidth(2)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout_5.addWidget(self.line)
        self.gridLayout_2.addLayout(self.verticalLayout_5, 1, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.tool_life_box.setTitle(_translate("MainWindow", "Tool-life"))
        self.wear.setText(_translate("MainWindow", "마모도"))
        self.wear_val.setText(_translate("MainWindow", "60 %"))
        self.cut_num.setText(_translate("MainWindow", "절단 횟수"))
        self.cut_num_val.setText(_translate("MainWindow", "90 회"))
        self.info_box.setTitle(_translate("MainWindow", "절단봉 정보"))
        self.rotate_val.setText(_translate("MainWindow", "////////////"))
        self.round_val.setText(_translate("MainWindow", "///////////"))
        self.rotate.setText(_translate("MainWindow", "- 회전"))
        self.round.setText(_translate("MainWindow", "- 둘레"))
        self.button_start.setText(_translate("MainWindow", "start"))
        self.stop.setText(_translate("MainWindow", "stop "))
        self.error_box.setTitle(_translate("MainWindow", "불량 검출"))
        self.tool_life.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:400;\">- 툴라이프 수명</span></p></body></html>"))

        green = QtGui.QPixmap('img/green.png')
        red = QtGui.QPixmap('img/red.png')
        green = green.scaledToWidth(13)
        red = red.scaledToWidth(13)
        self.tooler_error_val.setPixmap(red)
        self.matter_val.setPixmap(green)
        self.tool_life_val.setPixmap(green)


        self.tooler_error.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:400;\">- 절단봉 툴러 불량</span></p></body></html>"))
        self.matter.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:400;\">- 절단봉 이물질</span></p></body></html>"))
        self.ng_good.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:16pt;\">NG / GOOD</span></p></body></html>"))
        self.name.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:20pt;\">절단봉 불량 검출</span></p><p align=\"center\"><span style=\" font-size:20pt;\">프로그램</span></p></body></html>"))
        self.logo.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">LOGO</p></body></html>"))
        self.co_name.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt;\">COMPANY NAME</span></p></body></html>"))
        self.server.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">SERVER</p></body></html>"))
        self.camera.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">CAMERA</p></body></html>"))
        self.camera_val.setText(_translate("MainWindow", "카메라정보"))
        self.server_val.setText(_translate("MainWindow", "서버정보"))

    @pyqtSlot(str)
    def show_img(self, img):
        print('re')
        pixmap = QtGui.QPixmap(img)
        pixmap = pixmap.scaledToWidth(421)
        self.img.setPixmap(pixmap)

stylesheet = """
    QMainWindow {
        background-image: url("img/red.png"); 
        background-repeat: no-repeat; 
        background-position: center;
    }
"""

if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    example=Example(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())


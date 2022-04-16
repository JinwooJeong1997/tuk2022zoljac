from PyQt5.QtWidgets import  QWidget,QMessageBox,QDialog
import configparser
from PyQt5.QtCore import Qt
from PyQt5 import QtCore, QtGui, QtWidgets
from pypylon import pylon
import camera_connection as cc



#카메라 설정 화면 생성, 관리

class Camera(QWidget):

    def __init__(self, parent=None):
        super(self.__class__, self).__init__(parent)

    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(326, 214)
        Dialog.setMaximumSize(QtCore.QSize(430, 200))
        Dialog.setMinimumSize(QtCore.QSize(430, 200))
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.cancel = QtWidgets.QPushButton(Dialog)
        self.cancel.setMinimumSize(QtCore.QSize(0, 50))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.cancel.setFont(font)
        self.cancel.setObjectName("cancel")
        self.gridLayout.addWidget(self.cancel, 5, 2, 1, 1)
        self.connect = QtWidgets.QPushButton(Dialog)
        self.connect.setMinimumSize(QtCore.QSize(0, 50))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.connect.setFont(font)
        self.connect.setObjectName("connect")
        self.gridLayout.addWidget(self.connect, 5, 1, 1, 1)
        self.port = QtWidgets.QLabel(Dialog)
        self.port.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.port.setFont(font)
        self.port.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.port.setObjectName("port")
        self.gridLayout.addWidget(self.port, 2, 0, 1, 1)
        self.portVal = QtWidgets.QComboBox(Dialog)
        self.portVal.setObjectName("portVal")
        self.gridLayout.addWidget(self.portVal, 1, 1, 1, 1)
        self.method = QtWidgets.QLabel(Dialog)
        self.method.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.method.setFont(font)
        self.method.setObjectName("method")
        self.gridLayout.addWidget(self.method, 1, 0, 1, 1, QtCore.Qt.AlignRight)
        self.methodVal = QtWidgets.QComboBox(Dialog)
        self.methodVal.setObjectName("methodVal")
        self.gridLayout.addWidget(self.methodVal, 2, 1, 1, 1)

        self.current = QtWidgets.QLabel(Dialog)
        self.current.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.current.setFont(font)
        self.current.setObjectName("method")
        self.gridLayout.addWidget(self.current, 3, 0, 1, 1, QtCore.Qt.AlignRight)
        self.currentVal = QtWidgets.QLabel(Dialog)
        self.currentVal.setObjectName("methodVal")
        self.gridLayout.addWidget(self.currentVal, 3, 1, 1, 1)
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        camera = configparser.ConfigParser()
        camera.read('camera.ini', encoding='utf-8')
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.cancel.setText(_translate("Dialog", "연결 끊기"))
        self.connect.setText(_translate("Dialog", "카메라 연결"))
        self.port.setText(_translate("Dialog", "USB"))
        self.method.setText(_translate("Dialog", "연결방법"))
        self.dialog.setWindowTitle('camera')
        self.portVal.addItem('USB')
        self.portVal.addItem('EtherNet')
        camera = configparser.ConfigParser()
        camera.read('camera.ini', encoding='utf-8')
        self.methodVal.addItem(camera['USB']['var1'])
        self.methodVal.addItem(camera['USB']['var2'])
        self.methodVal.addItem(camera['USB']['var3'])
        self.current.setText(_translate("Dialog", "현재 설정값"))
        self.currentVal.setText(camera['SET']['var1']+' , '+camera['SET']['var2'])

    def camera(self):
        self.dialog = QDialog()
        self.dialog.setWindowModality(Qt.NonModal)
        self.setupUi(self.dialog)
        self._connectSignals()
        self.dialog.show()

    def change(self):
        self.methodVal.clear()
        camera = configparser.ConfigParser()
        camera.read('camera.ini', encoding='utf-8')
        self.port.setText(self.portVal.currentText())
        if self.portVal.currentText()=='USB':
            self.methodVal.addItem(camera['USB']['var1'])
            self.methodVal.addItem(camera['USB']['var2'])
            self.methodVal.addItem(camera['USB']['var3'])
        else:
            camera = configparser.ConfigParser()
            camera.read('camera.ini', encoding='utf-8')
            camera['STATE'] = {}
            for i in pylon.TlFactory.GetInstance().EnumerateDevices():
                num = 1
                camera['STATE'][str(num)] = i.GetIpAddress()
                num + 1
            with open('camera.ini', 'w', encoding='utf-8') as configfile:
                camera.write(configfile)
            for i in camera['STATE']:
                self.methodVal.addItem(camera['STATE'][str(i)])

    def connection(self):
        camera = configparser.ConfigParser()
        camera.read('camera.ini', encoding='utf-8')
        camera['SET']={}
        camera['SET']['var1'] = self.portVal.currentText()
        camera['SET']['var2'] = self.methodVal.currentText()
        with open('camera.ini', 'w', encoding='utf-8') as configfile:
            camera.write(configfile)
        if self.portVal.currentText()=='EtherNet':
            f=cc.ETHERNET_camera()
        else:
            f=cc.USB_camera()
        msg = QMessageBox()
        msg.setWindowTitle('server')
        if f==1:
            msg.setText('camera connected')
        else:
            msg.setText('camera NOT found')
        msg.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)

        msg.exec_()

    def canceled(self):
        msg = QMessageBox()
        msg.setWindowTitle('server')

        msg.setText('camera unconnected')

        msg.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)

        msg.exec_()

    def _connectSignals(self):
        self.portVal.currentIndexChanged.connect(self.change)
        self.connect.clicked.connect(self.connection)
        self.cancel.clicked.connect(self.canceled)


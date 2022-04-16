from PyQt5.QtWidgets import QApplication, QWidget,QMessageBox,QDialog,QLineEdit
import configparser
from PyQt5.QtCore import QObject,Qt
from PyQt5 import QtCore, QtGui, QtWidgets

import server_serial

#서버 설정

class Server(QWidget):

    def __init__(self, parent=None):
        super(self.__class__, self).__init__(parent)

    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(438, 338)
        Dialog.setMaximumSize(QtCore.QSize(430, 200))
        Dialog.setMinimumSize(QtCore.QSize(430, 200))
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.port = QtWidgets.QLabel(Dialog)
        self.port.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.port.setFont(font)
        self.port.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.port.setObjectName("port")
        self.gridLayout.addWidget(self.port, 0, 0, 1, 1)
        self.IP1 = QtWidgets.QLineEdit(Dialog)
        self.IP1.setAlignment(QtCore.Qt.AlignRight)
        self.IP1.setObjectName("IP1")
        self.gridLayout.addWidget(self.IP1, 0, 1, 1, 1)
        self.dot1 = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.dot1.setFont(font)
        self.dot1.setObjectName("dot1")
        self.gridLayout.addWidget(self.dot1, 0, 2, 1, 1)
        self.IP2 = QtWidgets.QLineEdit(Dialog)
        self.IP2.setAlignment(QtCore.Qt.AlignRight)
        self.IP2.setObjectName("IP2")
        self.gridLayout.addWidget(self.IP2, 0, 3, 1, 1)
        self.dot1 = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.dot1.setFont(font)
        self.dot1.setObjectName("dot1")
        self.gridLayout.addWidget(self.dot1, 0, 4, 1, 1)
        self.IP3 = QtWidgets.QLineEdit(Dialog)
        self.IP3.setAlignment(QtCore.Qt.AlignRight)
        self.IP3.setObjectName("IP3")
        self.gridLayout.addWidget(self.IP3, 0, 5, 1, 1)
        self.dot3 = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.dot3.setFont(font)
        self.dot3.setObjectName("dot3")
        self.gridLayout.addWidget(self.dot3, 0, 6, 1, 1)
        self.IP4 = QtWidgets.QLineEdit(Dialog)
        self.IP4.setAlignment(QtCore.Qt.AlignRight)
        self.IP4.setObjectName("IP4")
        self.gridLayout.addWidget(self.IP4, 0, 7, 1, 1)
        self.method = QtWidgets.QLabel(Dialog)
        self.method.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.method.setFont(font)
        self.method.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.method.setObjectName("method")
        self.gridLayout.addWidget(self.method, 1, 0, 1, 1)
        self.PORTNum = QtWidgets.QLineEdit(Dialog)
        self.PORTNum.setObjectName("PORTNum")
        self.PORTNum.setAlignment(QtCore.Qt.AlignRight)
        self.gridLayout.addWidget(self.PORTNum, 1, 1, 1, 1)
        self.connect = QtWidgets.QPushButton(Dialog)
        self.connect.setMinimumSize(QtCore.QSize(0, 50))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.connect.setFont(font)
        self.connect.setObjectName("connect")
        self.gridLayout.addWidget(self.connect, 5, 1, 1, 3)
        self.cancel = QtWidgets.QPushButton(Dialog)
        self.cancel.setMinimumSize(QtCore.QSize(0, 50))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.cancel.setFont(font)
        self.cancel.setObjectName("cancel")
        self.gridLayout.addWidget(self.cancel, 5, 5, 1, 3)

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

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        server = configparser.ConfigParser()
        server.read('server.ini', encoding='utf-8')
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.port.setText(_translate("Dialog", "SERVER IP"))
        self.dot1.setText(_translate("Dialog", "."))
        self.dot1.setText(_translate("Dialog", "."))
        self.dot3.setText(_translate("Dialog", "."))
        self.method.setText(_translate("Dialog", "PORT"))
        self.connect.setText(_translate("Dialog", "서버연결"))
        self.cancel.setText(_translate("Dialog", "연결 끊기"))
        self.dialog.setWindowTitle('SERVER')
        self.IP1.setText(server['INFO']['ip1'])
        self.IP2.setText(server['INFO']['ip2'])
        self.IP3.setText(server['INFO']['ip3'])
        self.IP4.setText(server['INFO']['ip4'])
        self.PORTNum.setText(server['INFO']['port'])
        self.current.setText(_translate("Dialog", "연결상태"))
        self.currentVal.setText(_translate("Dialog", "   연결안됨"))

    def server(self):
        self.dialog = QDialog()
        self.dialog.setWindowModality(Qt.NonModal)
        self.setupUi(self.dialog)
        self._connectSignals()
        self.dialog.show()

    #serial 통신
    def connection_serial(self):
        self.ser = server_serial.server_serial()
        self.ser.start_thread(1)
        
        msg = QMessageBox()
        msg.setWindowTitle('server')
        msg.setText('server serial connected')
        msg.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)
        msg.exec_()
    
        
    def connection(self):
        server = configparser.ConfigParser()

        # 설정파일 오브젝트 만들기
        IP=self.IP1.text()+'.'+self.IP2.text()+'.'+self.IP3.text()+'.'+self.IP4.text()
        server['INFO'] = {}
        server['INFO']['ip'] =IP
        server['INFO']['ip1']=self.IP1.text()
        server['INFO']['ip2'] = self.IP2.text()
        server['INFO']['ip3'] = self.IP3.text()
        server['INFO']['ip4'] = self.IP4.text()
        server['INFO']['port']=self.PORTNum.text()

        with open('server.ini', 'w', encoding='utf-8') as configfile:
            server.write(configfile)

        msg = QMessageBox()
        msg.setWindowTitle('server')
        msg.setText('server connected')
        msg.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)
        msg.exec_()

    def canceled(self):
        msg = QMessageBox()
        msg.setWindowTitle('server')
        msg.setText('server unconnected')

        msg.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)

        msg.exec_()


    def _connectSignals(self):
        self.connect.clicked.connect(self.connection_serial)
        self.cancel.clicked.connect(self.canceled)
from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QVBoxLayout, QApplication
from PyQt5.QtCore import QObject,pyqtSignal, pyqtSlot, QThread
import sys
import time

class Window(QWidget):

    def __init__(self):
        super().__init__()

        self.button_start = QPushButton('Start', self)
        self.button_cancel = QPushButton('Cancel', self)
        self.label_status = QLabel('status!!', self)

        layout = QVBoxLayout(self)
        layout.addWidget(self.button_start)
        layout.addWidget(self.button_cancel)
        layout.addWidget(self.label_status)

        self.setFixedSize(400, 200)

    @pyqtSlot(int)
    def updateStatus(self, status):
        self.label_status.setText('{}'.format(status))


class Example(QObject):

    def __init__(self, parent=None):
        super(self.__class__, self).__init__(parent)

        self.gui = Window()


        self.worker=Worker()
        self.worker_thread=QThread()
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.start()

        self._connectSignals()

        self.gui.show()


    def _connectSignals(self):
        self.gui.button_start.clicked.connect(self.worker.startWork)

        self.worker.sig_numbers.connect(self.gui.updateStatus)

        self.gui.button_cancel.clicked.connect(self.forceWorkerReset)

    def forceWorkerReset(self):
        if self.worker_thread.isRunning():
            self.worker_thread.terminate()
            self.worker_thread.wait()
            self.worker_thread.start()


class Worker(QObject):
    sig_numbers = pyqtSignal(int)

    def __init__(self, parent=None):
        super(self.__class__, self).__init__(parent)

    @pyqtSlot()
    def startWork(self):
        _cnt = 0
        while _cnt < 10:
            _cnt += 1
            self.sig_numbers.emit(_cnt)
            print(_cnt)
            time.sleep(1)




if __name__ == "__main__":
    app = QApplication(sys.argv)
    example = Example(app)
    sys.exit(app.exec_())

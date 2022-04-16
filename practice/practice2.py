from PyQt5.QtWidgets import *
import sys


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setupUI()

    def setupUI(self):
        self.setGeometry(200, 200, 200, 200)
        self.setWindowTitle("Click")

        self.OnOpenDocument_Button = QPushButton("File Open")
        self.OnOpenDocument_Button.clicked.connect(self.OnOpenDocument)
        self.label = QLabel()

        layout = QVBoxLayout()
        layout.addWidget(self.OnOpenDocument_Button)
        layout.addWidget(self.label)

        self.setLayout(layout)


    def OnOpenDocument(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', "",
                                        "All Files(*);; Python Files(*.py)", '/home')
        if fname[0]:
            print(type(fname[0]))
        else:
            QMessageBox.about(self, "Warning", "파일을 선택하지 않았습니다.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec_()
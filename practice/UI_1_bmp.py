# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'tool_layout_ui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget,QMessageBox,QFileDialog
import sys
from PyQt5.QtCore import QObject,pyqtSignal, pyqtSlot, QThread,Qt
import time
import math
import cv2
import numpy as np

import matplotlib.pyplot as plt



def grayscale(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img,low_threshold,high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def region_of_interest1_up(img,x,y,radius):
    square=int(0.13*radius)
    masked_image=img[int(y)-5*square:int(y)-4*square,int(x)+4*square:int(x)+5*square].copy()
    return masked_image

def region_of_interest2_up(img,x,y,radius):
    square = int(0.13 * radius)
    masked_image = img[int(y) - 5 * square:int(y) - 4 * square, int(x) + 3 * square:int(x) + 4 * square].copy()
    return masked_image

def region_of_interest3_up(img,x,y,radius):
    square=int(0.13*radius)
    masked_image=img[int(y)-4*square:int(y)-3*square,int(x)+4*square:int(x)+5*square].copy()
    return masked_image

def region_of_interest4_up(img,x,y,radius):
    square = int(0.13 * radius)
    masked_image = img[int(y) - 4 * square:int(y) - 3 * square, int(x) + 3 * square:int(x) + 4 * square].copy()
    return masked_image

def region_of_interest1_down(img,x,y,radius):
    square=int(0.13*radius)
    masked_image=img[int(y)+4*square:int(y)+5*square,int(x)+4*square:int(x)+5*square].copy()
    return masked_image

def region_of_interest2_down(img,x,y,radius):
    square = int(0.13 * radius)
    masked_image = img[int(y) +4 * square:int(y) +5 * square, int(x) + 3 * square:int(x) + 4 * square].copy()
    return masked_image

def region_of_interest3_down(img,x,y,radius):
    square=int(0.13*radius)
    masked_image=img[int(y)+3*square:int(y)+4*square,int(x)+4*square:int(x)+5*square].copy()
    return masked_image

def region_of_interest4_down(img,x,y,radius):
    square = int(0.13 * radius)
    masked_image = img[int(y) +3 * square:int(y) +4 * square, int(x) + 3 * square:int(x) + 4 * square].copy()
    return masked_image
def region_of_interest5_up(img,x,y,radius):
    square=int(0.13*radius)
    masked_image=img[int(y)-5*square:int(y)-4*square,int(x)-5*square:int(x)-4*square].copy()
    return masked_image

def region_of_interest6_up(img,x,y,radius):
    square = int(0.13 * radius)
    masked_image = img[int(y) - 5 * square:int(y) - 4 * square, int(x) + -4 * square:int(x) -3 * square].copy()
    return masked_image

def region_of_interest7_up(img,x,y,radius):
    square=int(0.13*radius)
    masked_image=img[int(y)-4*square:int(y)-3*square,int(x)-5*square:int(x)-4*square].copy()
    return masked_image

def region_of_interest8_up(img,x,y,radius):
    square = int(0.13 * radius)
    masked_image = img[int(y) - 4 * square:int(y) - 3 * square, int(x) -4 * square:int(x) -3 * square].copy()
    return masked_image

def region_of_interest5_down(img,x,y,radius):
    square=int(0.13*radius)
    masked_image=img[int(y)+4*square:int(y)+5*square,int(x)-5*square:int(x)-4*square].copy()
    return masked_image

def region_of_interest6_down(img,x,y,radius):
    square = int(0.13 * radius)
    masked_image = img[int(y) +4 * square:int(y) +5 * square, int(x) -4 * square:int(x) -3 * square].copy()
    return masked_image

def region_of_interest7_down(img,x,y,radius):
    square=int(0.13*radius)
    masked_image=img[int(y)+3*square:int(y)+4*square,int(x)-5*square:int(x)-4*square].copy()
    return masked_image

def region_of_interest8_down(img,x,y,radius):
    square = int(0.13 * radius)
    masked_image = img[int(y) +3 * square:int(y) +4 * square, int(x) -4 * square:int(x) -3 * square].copy()
    return masked_image


def region_of_interest0(img,x,y,radius):
    mask=np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,)
    else:
        ignore_mask_color = 255

    cv2.circle(mask, (int(x),int(y)), int(radius), (255, 255, 255), -1)
    masked_image=cv2.bitwise_and(img,mask)
    return masked_image

def calc_lines(lines):
    angles=[]
    ab=[]
    cnt90=0
    cnt0=0
    for line in lines:
        for x1,y1,x2,y2 in line:
            if x2-x1!=0 and y2-y1!=0:
                angle=(y1-y2)/(x2-x1)
                a = math.degrees(math.atan(angle))
                ab.append(abs(a))
                angles.append(a)
            elif x2-x1==0:
                cnt90+=1
            elif y2-y1==0:
                cnt0+=1
    if len(angles)==0:
        if cnt0>cnt90: mean=0
        elif cnt0<cnt90: mean=90

        else: mean='None'
    elif sum(ab)/len(ab)>85: mean=sum(ab)/len(ab)
    else:
        if sum(angles)/len(angles) >= 85:
            mean = (sum(angles) + 90 * cnt90) / (len(angles) + cnt90)
        elif sum(angles)/len(angles) >= 5:
            mean= sum(angles) / len(angles)
        elif sum(angles)/len(angles) >= -5:
            mean= sum(angles) / (len(angles)+cnt0)
        elif sum(angles)/len(angles) >= -85:
            mean= sum(angles) / len(angles)
        elif sum(angles)/len(angles) >= -90:
            mean= (sum(angles) - 90 * cnt90) / (len(angles) + cnt90)
        else: mean='None'

    return mean

def draw_lines(img, lines,color=[255,0,0],thickness=1):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img,(x1,y1),(x2,y2),color,thickness)

def hough_lines(img,rho,theta,threshold,min_line_len,max_line_gap):
    lines=cv2.HoughLinesP(img,rho,theta,threshold,np.array([]),
                          minLineLength=min_line_len,
                          maxLineGap=max_line_gap)
    if lines is not None:
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        a = calc_lines(lines)
        draw_lines(line_img, lines)
    else:
        a = 'None'
    return line_img, a

def hough_circle(img):
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1.2, 30, None, 200,500,3000)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # 원 둘레에 초록색 원 그리기
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # 원 중심점에 빨강색 원 그리기
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 5)
    else: print('no circle')
    return img

def adapt_thres(img):
    thr1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 127, 5)
    return thr1

def thresholding(img):
    ret,thr=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return thr

def contour(img):
    contours,_=cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    contour=contours[0]
    (x, y), radius = cv2.minEnclosingCircle(contour)
    return x,y,radius

def sharpening(img):
    sharpening_mask = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpening = cv2.filter2D(img, -1, sharpening_mask)
    return sharpening

def calc4(a,b,c,d):
    ab_arr1=[abs(a),abs(b),abs(c),abs(d)]
    arr1=[a,b,c,d]
    ab_sum1=0
    ab_cnt1=0
    cnt90_1=0
    sum1=0
    cnt1=0
    for i in ab_arr1:
        if i !='None':
            ab_sum1+=i
            ab_cnt1+=1

    ab=ab_sum1/ab_cnt1

    if ab>=88: avg1=ab
    else:
        for i in arr1:
            if i !='None' and i!=90:
                sum1+=i
                cnt1+=1
            elif i==90:
                cnt90_1+=1

        if sum1/cnt1-90<-170:
            avg1=(sum1-90*cnt90_1)/(cnt1+cnt90_1)
        else:
            avg1=(sum1+90*cnt90_1)/(cnt1+cnt90_1)

    return avg1

def masking(img,x,y,radius):
    mask1_up = region_of_interest1_up(img, x, y, radius)
    mask2_up = region_of_interest2_up(img, x, y, radius)
    mask3_up = region_of_interest3_up(img, x, y, radius)
    mask4_up = region_of_interest4_up(img, x, y, radius)
    mask1_d = region_of_interest1_down(img, x, y, radius)
    mask2_d = region_of_interest2_down(img, x, y, radius)
    mask3_d = region_of_interest3_down(img, x, y, radius)
    mask4_d = region_of_interest4_down(img, x, y, radius)
    mask5_up = region_of_interest5_up(img, x, y, radius)
    mask6_up = region_of_interest6_up(img, x, y, radius)
    mask7_up = region_of_interest7_up(img, x, y, radius)
    mask8_up = region_of_interest8_up(img, x, y, radius)
    mask5_d = region_of_interest5_down(img, x, y, radius)
    mask6_d = region_of_interest6_down(img, x, y, radius)
    mask7_d = region_of_interest7_down(img, x, y, radius)
    mask8_d = region_of_interest8_down(img, x, y, radius)

    mask1_up = sharpening(mask1_up)
    mask2_up = sharpening(mask2_up)
    mask3_up = sharpening(mask3_up)
    mask4_up = sharpening(mask4_up)
    mask1_d = sharpening(mask1_d)
    mask2_d = sharpening(mask2_d)
    mask3_d = sharpening(mask3_d)
    mask4_d = sharpening(mask4_d)
    mask5_up = sharpening(mask5_up)
    mask6_up = sharpening(mask6_up)
    mask7_up = sharpening(mask7_up)
    mask8_up = sharpening(mask8_up)
    mask5_d = sharpening(mask5_d)
    mask6_d = sharpening(mask6_d)
    mask7_d = sharpening(mask7_d)
    mask8_d = sharpening(mask8_d)

    apt1 = adapt_thres(mask1_up)
    apt2 = adapt_thres(mask2_up)
    apt3 = adapt_thres(mask3_up)
    apt4 = adapt_thres(mask4_up)
    apt5 = adapt_thres(mask1_d)
    apt6 = adapt_thres(mask2_d)
    apt7 = adapt_thres(mask3_d)
    apt8 = adapt_thres(mask4_d)
    apt9 = adapt_thres(mask5_up)
    apt10 = adapt_thres(mask6_up)
    apt11 = adapt_thres(mask7_up)
    apt12 = adapt_thres(mask8_up)
    apt13 = adapt_thres(mask5_d)
    apt14 = adapt_thres(mask6_d)
    apt15 = adapt_thres(mask7_d)
    apt16 = adapt_thres(mask8_d)

    # 라인검출
    rho = 2
    theta = np.pi / 180
    threshold = 300
    min_line_len = 100
    max_line_gap = 150
    lines1, a = hough_lines(apt1, rho, theta, threshold, min_line_len, max_line_gap)
    lines2, b = hough_lines(apt2, rho, theta, threshold, min_line_len, max_line_gap)
    lines3, c = hough_lines(apt3, rho, theta, threshold, min_line_len, max_line_gap)
    lines4, d = hough_lines(apt4, rho, theta, threshold, min_line_len, max_line_gap)
    lines5, e = hough_lines(apt5, rho, theta, threshold, min_line_len, max_line_gap)
    lines6, f = hough_lines(apt6, rho, theta, threshold, min_line_len, max_line_gap)
    lines7, g = hough_lines(apt7, rho, theta, threshold, min_line_len, max_line_gap)
    lines8, h = hough_lines(apt8, rho, theta, threshold, min_line_len, max_line_gap)
    lines9, i = hough_lines(apt9, rho, theta, threshold, min_line_len, max_line_gap)
    lines10, j = hough_lines(apt10, rho, theta, threshold, min_line_len, max_line_gap)
    lines11, k = hough_lines(apt11, rho, theta, threshold, min_line_len, max_line_gap)
    lines12, l = hough_lines(apt12, rho, theta, threshold, min_line_len, max_line_gap)
    lines13, m = hough_lines(apt13, rho, theta, threshold, min_line_len, max_line_gap)
    lines14, n = hough_lines(apt14, rho, theta, threshold, min_line_len, max_line_gap)
    lines15, o = hough_lines(apt15, rho, theta, threshold, min_line_len, max_line_gap)
    lines16, p = hough_lines(apt16, rho, theta, threshold, min_line_len, max_line_gap)
    sum_A = calc4(a, b, c, d)
    sum_B = calc4(e, f, g, h)
    sum_C = calc4(i, j, k, l)
    sum_D = calc4(m, n, o, p)
    print(a, b, c, d, sum_A)
    print(e, f, g, h, sum_B)
    print(i, j, k, l, sum_C)
    print(m, n, o, p, sum_D)


    return sum_A,sum_B,sum_C,sum_D


class Circle(QWidget):
    sig=pyqtSignal(str,int,int)

    def __init__(self,parent=None):
        super(self.__class__, self).__init__(parent)

    def OnOpenDocument(self):
        fname = QFileDialog.getOpenFileName()
        if fname[0]:
            print(fname[0])
            self.circle_analy(fname[0])
        else:
            QMessageBox.about(self, "Warning", "파일을 선택하지 않았습니다.")


    @pyqtSlot()
    def circle_analy(self,name):
        start = time.time()  # 시작 시간 저장

        # 작업 코드


        img_name=name
        img = cv2.imread(img_name)
        gray = grayscale(img)

        # 원 영역 검출
        kernel_size = 331
        blur_gray = gaussian_blur(gray, kernel_size)
        apt = thresholding(blur_gray)
        print("time :", time.time() - start)
        self.x, self.y, self.radius = contour(apt)
        print(self.x, self.y, self.radius)
        print("time :", time.time() - start)
        #마스크
        sum_A,sum_B,sum_C,sum_D=masking(gray,self.x,self.y,self.radius)
        rotate=0
        if int(sum_A)*int(sum_B)<0 or abs(abs(sum_A)-abs(sum_B))<=3 or abs(sum_A)>85 or abs(sum_B)>85:
            height, width = gray.shape
            matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 30, 1)
            dst = cv2.warpAffine(gray, matrix, (width, height))
            sum_A, sum_B, sum_C, sum_D = masking(dst, self.x, self.y, self.radius)
            avg = (sum_A + sum_B + sum_C + sum_D) / 4
            if abs(sum_B)+abs(sum_D)>=abs(sum_A)+abs(sum_C): rotate=avg-30
            else: rotate=180+avg-30

        else:
            avg=(sum_A+sum_B+sum_C+sum_D)/4
            if abs(sum_B)+abs(sum_D)>=abs(sum_A)+abs(sum_C): rotate=avg
            else: rotate=180+avg
        print("time :", time.time() - start)
        self.sig.emit(img_name,int(self.radius),int(rotate))

class Test2(QObject):


    def __init__(self, parent=None):
        super(self.__class__, self).__init__(parent)

    def camera(self):
        msg = QMessageBox()
        msg.setWindowTitle('camera')

        msg.setText('camera infomation')

        msg.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)

        msg.exec_()



    def server(self):
        msg = QMessageBox()
        msg.setWindowTitle('server')

        msg.setText('server infomation')

        msg.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)

        msg.exec_()




class Example(QObject):

    def __init__(self, parent=None):
        super(self.__class__, self).__init__(parent)

        self.gui = Ui_MainWindow()

        self.thread = QThread()
        self.thread.start()
        self.test = Circle()
        self.test.moveToThread(self.thread)

        self.thread2 = QThread()
        self.thread2.start()
        self.test2 = Test2()
        self.gui.moveToThread(self.thread2)



        self.gui.setupUi(MainWindow)

        self._connectSignals()


    def _connectSignals(self):
        self.gui.start.clicked.connect(self.test.OnOpenDocument)
        self.test.sig.connect(self.gui.show_img)
        self.gui.camera_info.clicked.connect(self.test2.camera)
        self.gui.server_info.clicked.connect(self.test2.server)



    def forceWorkerReset(self):
        if self.worker_thread.isRunning():
            self.worker_thread.terminate()
            self.worker_thread.wait()
            self.worker_thread.start()


class Ui_MainWindow(QWidget):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowTitle("")
        MainWindow.resize(1000, 900)
        MainWindow.setStyleSheet("background-color: #ffffff")
        MainWindow.setMaximumSize(QtCore.QSize(1000,900))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.tool_life_box = QtWidgets.QGroupBox(self.centralwidget)
        self.tool_life_box.setMinimumSize(QtCore.QSize(300, 300))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)

        self.tool_life_box.setFont(font)
        self.tool_life_box.setAlignment(QtCore.Qt.AlignCenter)
        self.tool_life_box.setObjectName("tool_life_box")
        self.tool_life_box.setMinimumSize(QtCore.QSize(150, 360))
        self.tool_img = QtWidgets.QLabel(self.tool_life_box)
        self.tool_img.setGeometry(QtCore.QRect(94, 20, 360, 360))
        self.tool_img.setMinimumSize(QtCore.QSize(0, 0))
        self.tool_img.setMaximumSize(QtCore.QSize(1000, 1000))
        font = QtGui.QFont()
        font.setPointSize(23)
        font.setBold(True)
        font.setWeight(75)
        self.tool_img.setFont(font)
        self.tool_img.setText("")
        self.tool_img.setObjectName("tool_img")
        self.cut_num = QtWidgets.QLabel(self.tool_life_box)
        self.cut_num.setGeometry(QtCore.QRect(233, 170, 51, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(23)
        font.setBold(True)
        font.setWeight(75)
        self.cut_num.setFont(font)
        self.cut_num.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.cut_num.setObjectName("cut_num")
        self.wear = QtWidgets.QLabel(self.tool_life_box)
        self.wear.setGeometry(QtCore.QRect(232, 245, 53, 30))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(23)
        font.setBold(True)
        font.setWeight(75)
        self.wear.setFont(font)
        self.wear.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.wear.setObjectName("wear")
        self.label = QtWidgets.QLabel(self.tool_life_box)
        self.label.setGeometry(QtCore.QRect(200, 130, 140, 41))
        self.label.setObjectName("label")
        self.label.setFont(font)
        self.label_2 = QtWidgets.QLabel(self.tool_life_box)
        self.label_2.setGeometry(QtCore.QRect(220, 205, 101, 41))
        self.label_2.setObjectName("label_2")
        self.label_2.setFont(font)

        self.gridLayout_5.addWidget(self.tool_life_box, 0, 0, 1, 1)
        self.info_box = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.info_box.setFont(font)
        font.setPointSize(30)
        self.info_box.setAlignment(QtCore.Qt.AlignCenter)
        self.info_box.setObjectName("info_box")
        self.gridLayout_10 = QtWidgets.QGridLayout(self.info_box)
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.start = QtWidgets.QPushButton(self.info_box)
        self.start.setFont(font)
        self.start.setObjectName("start")
        self.gridLayout_10.addWidget(self.start, 1, 0, 1, 1)
        self.gridLayout_9 = QtWidgets.QGridLayout()
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.rotate_val = QtWidgets.QLabel(self.info_box)
        self.rotate_val.setFont(font)
        self.rotate_val.setObjectName("rotate_val")
        self.gridLayout_9.addWidget(self.rotate_val, 1, 1, 1, 1)
        self.round_val = QtWidgets.QLabel(self.info_box)
        self.round_val.setObjectName("round_val")
        self.round_val.setFont(font)
        self.gridLayout_9.addWidget(self.round_val, 0, 1, 1, 1)
        self.rotate = QtWidgets.QLabel(self.info_box)
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(30)
        font.setBold(True)
        font.setWeight(75)
        self.rotate.setFont(font)
        self.rotate.setAlignment(QtCore.Qt.AlignCenter)
        self.rotate.setObjectName("rotate")
        self.gridLayout_9.addWidget(self.rotate, 1, 0, 1, 1)
        self.round = QtWidgets.QLabel(self.info_box)
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(30)
        font.setBold(True)
        font.setWeight(75)
        self.round.setFont(font)
        self.round.setAlignment(QtCore.Qt.AlignCenter)
        self.round.setObjectName("round")
        self.gridLayout_9.addWidget(self.round, 0, 0, 1, 1)
        self.gridLayout_10.addLayout(self.gridLayout_9, 0, 0, 1, 1)
        self.gridLayout_5.addWidget(self.info_box, 0, 1, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout_5, 3, 0, 1, 1)
        self.gridlayout = QtWidgets.QGridLayout()
        self.gridlayout.setContentsMargins(-1, -1, 0, -1)
        self.gridlayout.setSpacing(6)
        self.gridlayout.setObjectName("gridlayout")
        self.error_box = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(14)
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
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.tool_life.setFont(font)
        self.tool_life.setObjectName("tool_life")
        self.gridLayout_11.addWidget(self.tool_life, 2, 0, 1, 1)
        self.tooler_error_val = QtWidgets.QLabel(self.error_box)
        self.tooler_error_val.setMaximumSize(QtCore.QSize(200, 50))
        self.tooler_error_val.setAlignment(QtCore.Qt.AlignCenter)
        self.tooler_error_val.setText("")
        self.tooler_error_val.setObjectName("tooler_error_val")
        self.gridLayout_11.addWidget(self.tooler_error_val, 0, 1, 1, 1)
        self.matter_val = QtWidgets.QLabel(self.error_box)
        self.matter_val.setMaximumSize(QtCore.QSize(5000, 50))
        self.matter_val.setText("")
        self.matter_val.setObjectName("matter_val")
        self.matter_val.setAlignment(QtCore.Qt.AlignCenter)
        self.gridLayout_11.addWidget(self.matter_val, 1, 1, 1, 1)
        self.tool_life_val = QtWidgets.QLabel(self.error_box)
        self.tool_life_val.setMaximumSize(QtCore.QSize(5000, 50))
        self.tool_life_val.setText("")
        self.tool_life_val.setObjectName("tool_life_val")
        self.tool_life_val.setAlignment(QtCore.Qt.AlignCenter)
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
        font.setPointSize(30)
        font.setBold(True)
        font.setWeight(75)
        self.ng_good.setFont(font)
        self.ng_good.setObjectName("ng_good")
        self.ng_good.setMaximumSize(QtCore.QSize(500, 130))
        self.gridLayout_4.addWidget(self.ng_good, 0, 0, 1, 1)
        self.gridLayout_6.addLayout(self.gridLayout_4, 0, 0, 1, 1)
        self.gridlayout.addWidget(self.error_box, 0, 1, 1, 1)
        self.img = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(100)
        self.img.setFont(font)
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
        font.setPointSize(30)
        font.setBold(True)
        font.setWeight(75)
        self.name.setFont(font)
        self.name.setObjectName("name")
        self.name.setAlignment(QtCore.Qt.AlignCenter)
        self.verticalLayout_9.addWidget(self.name)
        self.horizontalLayout_2.addLayout(self.verticalLayout_9)
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.logo = QtWidgets.QLabel(self.centralwidget)
        self.logo.setMinimumSize(QtCore.QSize(352, 0))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(48)
        font.setBold(False)
        font.setWeight(50)
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
        self.gridLayout_3.setContentsMargins(-1, -1, 0, -1)
        self.gridLayout_3.setSpacing(6)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.camera_rg = QtWidgets.QLabel(self.centralwidget)
        self.camera_rg.setMaximumSize(QtCore.QSize(50, 50))
        self.camera_rg.setText("")
        self.camera_rg.setObjectName("camera_rg")
        self.gridLayout_3.addWidget(self.camera_rg, 1, 1, 1, 1)
        self.server_rg = QtWidgets.QLabel(self.centralwidget)
        self.server_rg.setMaximumSize(QtCore.QSize(50, 50))
        self.server_rg.setText("")
        self.server_rg.setObjectName("server_rg")
        self.gridLayout_3.addWidget(self.server_rg, 0, 1, 1, 1)
        self.server = QtWidgets.QLabel(self.centralwidget)
        self.server.setMaximumSize(QtCore.QSize(100, 300))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(30)
        font.setBold(True)
        font.setWeight(75)
        self.server.setFont(font)
        self.server.setAlignment(QtCore.Qt.AlignCenter)
        self.server.setObjectName("server")
        self.gridLayout_3.addWidget(self.server, 0, 0, 1, 1)
        self.camera = QtWidgets.QLabel(self.centralwidget)
        self.camera.setMaximumSize(QtCore.QSize(250, 300))
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setPointSize(30)
        font.setBold(True)
        font.setWeight(75)
        self.camera.setFont(font)
        self.camera.setAlignment(QtCore.Qt.AlignCenter)
        self.camera.setObjectName("camera")
        self.gridLayout_3.addWidget(self.camera, 1, 0, 1, 1)
        self.server_info = QtWidgets.QPushButton(self.centralwidget)
        self.server_info.setMaximumSize(QtCore.QSize(250, 300))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.server_info.setFont(font)
        self.server_info.setObjectName("server_info")
        self.server_info.setStyleSheet("background-color: #dddddd")
        self.gridLayout_3.addWidget(self.server_info, 0, 2, 1, 1)
        self.camera_info = QtWidgets.QPushButton(self.centralwidget)
        self.camera_info.setStyleSheet("background-color: #dddddd")
        self.camera_info.setMinimumSize(QtCore.QSize(160,50))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.camera_info.setFont(font)
        self.camera_info.setObjectName("camera_info")
        self.gridLayout_3.addWidget(self.camera_info, 1, 2, 1, 1)
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
        MainWindow.setWindowTitle(_translate("MainWindow", "절단봉 불량검출 프로그램"))
        MainWindow.setWindowIcon(QtGui.QIcon('img/logo.png'))
        self.tool_life_box.setTitle(_translate("MainWindow", "Tool-life"))
        self.cut_num.setText(_translate("MainWindow", "90"))
        self.wear.setText(_translate("MainWindow", "80"))
        self.label.setText(_translate("MainWindow", "절단 횟수"))
        self.label_2.setText(_translate("MainWindow", "마모도"))
        self.info_box.setTitle(_translate("MainWindow", "절단봉 정보"))
        self.start.setText(_translate("MainWindow", "file input"))
        self.rotate_val.setText(_translate("MainWindow", "90   도"))
        self.round_val.setText(_translate("MainWindow", "200   mm"))
        self.rotate.setText(_translate("MainWindow", "- 회전"))
        self.round.setText(_translate("MainWindow", "- 둘레"))
        self.error_box.setTitle(_translate("MainWindow", "불량 검출"))
        self.tool_life.setText(_translate("MainWindow", "<html><head/><body><p align=\"left\"><span style=\" font-size:30pt;\">툴라이프 수명</span></p></body></html>"))
        self.tooler_error.setText(_translate("MainWindow", "<html><head/><body><p align=\"left\"><span style=\" font-size:30pt;\">절단봉 툴러 불량</span></p></body></html>"))
        self.matter.setText(_translate("MainWindow", "<html><head/><body><p align=\"left\"><span style=\" font-size:30pt;\">절단봉 이물질</span></p></body></html>"))
        self.ng_good.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:45pt;\">NG / GOOD</span></p></body></html>"))
        self.name.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">절단봉 불량 검출</p><p align=\"center\">프로그램</p></body></html>"))
        pixmap = QtGui.QPixmap('img/logo.PNG')
        pixmap = pixmap.scaledToWidth(120)
        self.logo.setPixmap(pixmap)
        self.logo.setAlignment(Qt.AlignHCenter)
        pixmap_co = QtGui.QPixmap('img/samkwang.PNG')
        pixmap_co = pixmap_co.scaledToHeight(35)
        self.co_name.setPixmap(pixmap_co)
        self.co_name.setAlignment(Qt.AlignHCenter)
        self.server.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:20pt;\">SERVER</span></p></body></html>"))
        self.camera.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:20pt;\">CAMERA</span></p></body></html>"))
        self.server_info.setText(_translate("MainWindow", "서버정보"))
        self.camera_info.setText(_translate("MainWindow", "카메라정보"))
        green = QtGui.QPixmap('img/green.png')
        red = QtGui.QPixmap('img/red.png')
        green1 = green.scaledToWidth(30)
        red1 = red.scaledToWidth(30)
        green2 = green.scaledToWidth(50)
        red2 = red.scaledToWidth(50)
        self.server_rg.setPixmap(green1)
        self.server_rg.setAlignment(Qt.AlignVCenter)
        self.camera_rg.setPixmap(green1)
        self.camera_rg.setAlignment(Qt.AlignVCenter)
        self.matter_val.setPixmap(green2)
        self.tool_life_val.setPixmap(red2)
        self.tooler_error_val.setPixmap(red2)
        self.ng_good.setStyleSheet("background-color: #ff0000")
        tool = QtGui.QPixmap('img/tool.PNG')
        tool = tool.scaledToWidth(340)
        self.tool_img.setPixmap(tool)
        pixmap_m = QtGui.QPixmap('img/0_0.jpg')
        pixmap_m = pixmap_m.scaledToWidth(540)
        self.img.setPixmap(pixmap_m)
        self.start.setStyleSheet("background-color: #dddddd")
        #self.name.setStyleSheet("background-color: #dfdffd")

    @pyqtSlot(str,int,int)
    def show_img(self, img,radius,rotate):
        pixmap_m = QtGui.QPixmap(img)
        pixmap_m = pixmap_m.scaledToWidth(540)
        self.img.setPixmap(pixmap_m)
        _translate = QtCore.QCoreApplication.translate
        self.round_val.setText('{}mm'.format(radius))
        self.rotate_val.setText('{}°'.format(rotate))





if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    example=Example(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())


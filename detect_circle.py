from PyQt5.QtWidgets import  QWidget,QMessageBox,QFileDialog
from PyQt5.QtCore import pyqtSignal, pyqtSlot
import configparser
import time
import cv2
import header as h
import Classification as c
import imageio
import os
import detect as yd


import serial

# 시리얼 포트 연결
port = "COM3"
baud = 9600
ser = serial.Serial(port,baud,timeout=1)


#원 검출, 영상처리



class Circle(QWidget):
    sig=pyqtSignal(str,int,int,int,str)
    sig2=pyqtSignal(str,bool,float)
    
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
        
        img = h.raw_to_jpg(img_name)
        print("trans_time :", time.time() - start)
        gray = h.grayscale(img)
        
        # 원 영역 검출

        kernel_size = 831
        blur_gray = h.gaussian_blur(gray, kernel_size)
        print("blur_time:", time.time()-start)
        apt = h.thresholding(blur_gray)
        print("apt_time :", time.time() - start)
        self.x, self.y, self.radius = h.contour(apt)
        print("radius value {}".format(self.radius))
        print(self.x, self.y, self.radius)
        self.T_radius=0.005*self.radius
        print("contour_time :", time.time() - start)
        #마스크
        sum_A,sum_B,sum_C,sum_D=h.masking(gray,self.x,self.y,self.radius)
        rotate=0
        if sum_A is None or sum_B is None or sum_C is None or sum_D is None:
            rotate='9999999'
        elif int(sum_A)*int(sum_B)<0 or abs(abs(sum_A)-abs(sum_B))<=3 or abs(sum_A)>80 or abs(sum_B)>80:
            height, width = gray.shape
            matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 45, 1)
            dst = cv2.warpAffine(gray, matrix, (width, height))
            sum_A, sum_B, sum_C, sum_D = h.masking(dst, self.x, self.y, self.radius)
            avg = (sum_A + sum_B + sum_C + sum_D) / 4
            if abs(sum_B)+abs(sum_D)>=abs(sum_A)+abs(sum_C): rotate=avg-45
            else: rotate=180+avg-45

        else:
            avg=(sum_A+sum_B+sum_C+sum_D)/4
            if abs(sum_B)+abs(sum_D)>=abs(sum_A)+abs(sum_C): rotate=avg
            else: rotate=180+avg
        print("time :", time.time() - start)

        dir_name='detect'
        output_path='outputs/%s' %(dir_name)
        uniq=1
        while os.path.exists(output_path):
            output_path='outputs/%s(%d)' %(dir_name,uniq)
            uniq+=1
        os.makedirs(output_path)

        rgb=h.rotate(gray,int(rotate))
        imageio.imsave(output_path+'/rotate.jpg', rgb)
        imageio.imsave(output_path+'/orginal.jpg',gray)
        os.makedirs(output_path + '/matter')
        os.makedirs(output_path + '/mamo')
        h.region_of_interest_mamo(self.x,output_path)
        matter_loc,arr=c.classifcation(output_path)
        h.image_change(matter_loc,self.x,output_path)
        m = str(max(arr))
        circle = configparser.ConfigParser()
        circle['INFO'] = {}
        circle['INFO']['반지름'] = str(int(self.T_radius)) + 'cm'
        circle['INFO']['둘레'] = str(int(self.T_radius * 2 * 3.14)) + 'cm'
        circle['INFO']['회전각도'] = str(int(rotate)) + '도'
        circle['INFO']['절단횟수'] = '30회'
        circle['INFO']['이물질 위치']=str(matter_loc)
        circle['INFO']['마모도']=str(arr)
        with open(output_path+'/circle.ini', 'w', encoding='utf-8') as configfile:
            circle.write(configfile)

        isgood = False
        
        
        
        restime = 100.0
        
        
        
        if len(matter_loc) > 0  or  m >= 90 : #이물질존재, 마모도90이상
            isgood = False
        
        if isgood is False :
            #print("defect!")
            #alarm
            
            start = time.time()
            ser.write(b'a')
            data=ser.readline().decode('ascii')
            end = time.time()
            resttime = end - start
        else :
            start = time.time()
            ser.write(b'b')
            data=ser.readline().decode('ascii')
            end = time.time()
            resttime = end - start
            
        resttime = round(resttime,5)
        resttime = resttime * 1000
       #self.sig.emit(output_path,int(self.T_radius),int(rotate),int(uniq),str(m))
        self.sig2.emit(output_path,bool(isgood),float(resttime))
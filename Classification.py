import cv2
import header as h
from yolov5 import detect as yd
import time
import os
import sys
def classifcation(p):
    img=cv2.imread(p+'/rotate.jpg')
    number=0
    x = 3012
    y = 2012
    radius = 1850
    mask = h.region_of_interest_white(img, x, y, radius - 150)
    gray = h.grayscale(mask)
    apt = h.thresholding_1(gray)
    #h.basic_showImg(apt)

    a=[]
    for i in range(1,4024,5):
        for j in range(1,6024,5):
            if apt[i][j]==0:
                a.append([i,j])
    print(a)
    b=[]
    if len(a)!=0:
        b.append([a[0][1],a[0][0]])
        for i in range(1,len(a)):
            if abs(a[i-1][0]-a[i][0])>100:
                b.append([a[i][1], a[i][0]])
            else:
                pass

        for i in b:
            roi=h.region_of_interest_make(img,i[0],i[1])
            cv2.imwrite(p+'/matter/'+str(number)+'.jpg', roi)
            print('완료')
            number+=1
    else: print('None')

    yd.detect(source='./'+p+'/matter', project='./'+p+'/matter')
    arr=yd.detect1(source='./'+p+'/mamo', project='./'+p+'/mamo')
    ar=[]
    ar1=[]
    for i in arr:
        ar.append(i.split())
    for i in range(len(ar)):
        ar[i][2]=ar[i][2].replace(",","")
        print(ar[i][2])
        ar1.append(int(ar[i][2]))
    return b,ar1



#classifcation()
'''
number=1
for root, dirs, files in os.walk('img/test2'):
    for fname in files:
        full_fname = os.path.join(root, fname)
        number=classifcation(full_fname,number)
'''


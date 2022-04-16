import cv2
import header as h
import time
import os
class Circle():


    def circle_analy(self,name):
        start = time.time()
        img_name=name
        img = h.raw_to_jpg(img_name)
        self.gray = h.grayscale(img)

        # 원 영역 검출
        '''
        kernel_size = 831
        blur_gray = h.gaussian_blur(self.gray, kernel_size)
        apt = h.thresholding(blur_gray)
        print("time :", time.time() - start)
        self.x, self.y, self.radius = h.contour(apt)
        
        print(self.x, self.y, self.radius)
        '''
        self.x=3000
        self.y=2000
        self.radius=1900
        print("time :", time.time() - start)
        #마스크
        sum_A,sum_B,sum_C,sum_D=h.masking(self.gray,self.x,self.y,self.radius)
        rotate=0
        if sum_A is None or sum_B is None or sum_C is None or sum_D is None:
            rotate='9999999'
        elif int(sum_A)*int(sum_B)<0 or abs(abs(sum_A)-abs(sum_B))<=4 or abs(sum_A)>80 or abs(sum_B)>80:
            height, width = self.gray.shape
            matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 45, 1)
            dst = cv2.warpAffine(self.gray, matrix, (width, height))
            sum_A, sum_B, sum_C, sum_D = h.masking(dst, self.x, self.y, self.radius)
            avg = (sum_A + sum_B + sum_C + sum_D) / 4
            if abs(sum_B)+abs(sum_D)>=abs(sum_A)+abs(sum_C): rotate=avg-45
            else: rotate=180+avg-45

        else:
            avg=(sum_A+sum_B+sum_C+sum_D)/4
            if abs(sum_B)+abs(sum_D)>=abs(sum_A)+abs(sum_C): rotate=avg
            else: rotate=180+avg
        print("time :", time.time() - start)
        if rotate!='999999':
            rotate1=self.re(rotate,self.radius,self.x,self.y)
        else: rotate1='999999'
        return rotate,rotate1

    def re(self,rotate,round,x,y):
        gray=self.gray
        height, width = gray.shape
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), -int(rotate), 1)
        dst = cv2.warpAffine(gray, matrix, (width, height))

        sum_A, sum_B, sum_C, sum_D = h.masking(dst, x, y, round)
        rotate = 0
        if int(sum_A) * int(sum_B) < 0 or abs(abs(sum_A) - abs(sum_B)) <= 4 or abs(sum_A) > 80 or abs(sum_B) > 80:
            height, width = gray.shape
            matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 45, 1)
            dst = cv2.warpAffine(dst, matrix, (width, height))
            sum_A, sum_B, sum_C, sum_D = h.masking(dst, x, y, round)
            avg = (sum_A + sum_B + sum_C + sum_D) / 4
            if abs(sum_B) + abs(sum_D) >= abs(sum_A) + abs(sum_C):
                rotate = avg - 45
            else:
                rotate = 180 + avg - 45

        else:
            avg = (sum_A + sum_B + sum_C + sum_D) / 4
            if abs(sum_B) + abs(sum_D) >= abs(sum_A) + abs(sum_C):
                rotate = avg
            else:
                rotate = 180 + avg

        return rotate

    def number(self, name):
        start = time.time()
        img_name = name
        img = h.raw_to_jpg(img_name)
        self.gray = h.grayscale(img)

        self.x = 3000
        self.y = 2000
        self.radius = 1900
        print("time :", time.time() - start)
        # 마스크
        ma,mi,av = h.num_masking(self.gray, self.x, self.y, self.radius)

        return ma,mi,av


def file_input(list,list1):

    f = open("result.txt", 'w')
    data1 = list
    data2 =list1
    f.write('회전  :')
    for i in range(len(data1)):
        f.write(str(data1[i])+',')
    f.write('차 :')
    for i in range(len(data2)):
        f.write(str(data2[i]) + ',')
    f.close()


result=[]
clfi=[]

M=[]
m=[]
A=[]
circle=Circle()
f = open("maxmin0_b.txt", 'w')
i=0

for root, dirs, files in os.walk('img/test2/0'):
    for fname in files:
        full_fname = os.path.join(root, fname)
        ma,mi,av=circle.number(full_fname)
        M.append(ma)
        m.append(mi)
        A.append(av)
        print('최댓값 평균:',(sum(M)/len(M)))
        print('최솟값 평균:',(sum(m)/len(m)))
        print('현재 평균:',(sum(A)/len(A)))

        f.write(str(ma)+str(mi)+str(av)+ '\n')
f.write('최종최댓값평균:'+str(sum(M)/len(M))+'최종최솟값평균:'+str(sum(m)/len(m))+'최종평균평균:'+str(sum(A)/len(A))+'최댓값최댓값'+str(max(M))+'최댓값최솟값'+str(min(M))+'최솟값최댓값'+str(max(m))+'최솟값최솟값'+str(min(m))+'평균최댓값'+str(max(A))+'평균최솟값'+str(min(A)))
f.close()

M=[]
m=[]
A=[]
circle=Circle()
f = open("maxmin25_b.txt", 'w')
i=0

for root, dirs, files in os.walk('img/test2/25'):
    for fname in files:
        full_fname = os.path.join(root, fname)
        ma,mi,av=circle.number(full_fname)
        M.append(ma)
        m.append(mi)
        A.append(av)
        print('최댓값 평균:',(sum(M)/len(M)))
        print('최솟값 평균:',(sum(m)/len(m)))
        print('현재 평균:',(sum(A)/len(A)))

        f.write(str(ma)+str(mi)+str(av)+ '\n')
f.write('최종최댓값평균:'+str(sum(M)/len(M))+'최종최솟값평균:'+str(sum(m)/len(m))+'최종평균평균:'+str(sum(A)/len(A))+'최댓값최댓값'+str(max(M))+'최댓값최솟값'+str(min(M))+'최솟값최댓값'+str(max(m))+'최솟값최솟값'+str(min(m))+'평균최댓값'+str(max(A))+'평균최솟값'+str(min(A)))
f.close()

M=[]
m=[]
A=[]
circle=Circle()
f = open("maxmin50_b.txt", 'w')
i=0

for root, dirs, files in os.walk('img/test2/50'):
    for fname in files:
        full_fname = os.path.join(root, fname)
        ma,mi,av=circle.number(full_fname)
        M.append(ma)
        m.append(mi)
        A.append(av)
        print('최댓값 평균:',(sum(M)/len(M)))
        print('최솟값 평균:',(sum(m)/len(m)))
        print('현재 평균:',(sum(A)/len(A)))

        f.write(str(ma)+str(mi)+str(av)+ '\n')
f.write('최종최댓값평균:'+str(sum(M)/len(M))+'최종최솟값평균:'+str(sum(m)/len(m))+'최종평균평균:'+str(sum(A)/len(A))+'최댓값최댓값'+str(max(M))+'최댓값최솟값'+str(min(M))+'최솟값최댓값'+str(max(m))+'최솟값최솟값'+str(min(m))+'평균최댓값'+str(max(A))+'평균최솟값'+str(min(A)))
f.close()

M=[]
m=[]
A=[]
circle=Circle()
f = open("maxmin75_b.txt", 'w')
i=0

for root, dirs, files in os.walk('img/test2/75'):
    for fname in files:
        full_fname = os.path.join(root, fname)
        ma,mi,av=circle.number(full_fname)
        M.append(ma)
        m.append(mi)
        A.append(av)
        print('최댓값 평균:',(sum(M)/len(M)))
        print('최솟값 평균:',(sum(m)/len(m)))
        print('현재 평균:',(sum(A)/len(A)))

        f.write(str(ma)+str(mi)+str(av)+ '\n')
f.write('최종최댓값평균:'+str(sum(M)/len(M))+'최종최솟값평균:'+str(sum(m)/len(m))+'최종평균평균:'+str(sum(A)/len(A))+'최댓값최댓값'+str(max(M))+'최댓값최솟값'+str(min(M))+'최솟값최댓값'+str(max(m))+'최솟값최솟값'+str(min(m))+'평균최댓값'+str(max(A))+'평균최솟값'+str(min(A)))
f.close()

M=[]
m=[]
A=[]
circle=Circle()
f = open("maxmin90_b.txt", 'w')
i=0

for root, dirs, files in os.walk('img/test2/90'):
    for fname in files:
        full_fname = os.path.join(root, fname)
        ma,mi,av=circle.number(full_fname)
        M.append(ma)
        m.append(mi)
        A.append(av)
        print('최댓값 평균:',(sum(M)/len(M)))
        print('최솟값 평균:',(sum(m)/len(m)))
        print('현재 평균:',(sum(A)/len(A)))

        f.write(str(ma)+str(mi)+str(av)+ '\n')
f.write('최종최댓값평균:'+str(sum(M)/len(M))+'최종최솟값평균:'+str(sum(m)/len(m))+'최종평균평균:'+str(sum(A)/len(A))+'최댓값최댓값'+str(max(M))+'최댓값최솟값'+str(min(M))+'최솟값최댓값'+str(max(m))+'최솟값최솟값'+str(min(m))+'평균최댓값'+str(max(A))+'평균최솟값'+str(min(A)))
f.close()
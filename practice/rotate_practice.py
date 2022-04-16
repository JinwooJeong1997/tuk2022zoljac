import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import rawpy
import imageio


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

def hough_lines1(img,rho,theta,threshold):
    lines = cv2.HoughLines(img, rho, theta, threshold)
    for line in lines:  # 검출된 모든 선 순회
        r, theta = line[0]  # 거리와 각도
        tx, ty = np.cos(theta), np.sin(theta)  # x, y축에 대한 삼각비
        x0, y0 = tx * r, ty * r  # x, y 기준(절편) 좌표
        print(math.degrees(theta))

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
    thr1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 2)
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
    sharpening_mask1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    blurring_mask1 = np.array([[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]])
    blurring_mask2 = np.array([[1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25], [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25],
                               [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25], [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25],
                               [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25]])
    sharpening_mask2 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    #sharpening = cv2.filter2D(img, -1, blurring_mask2)
    sharpening = cv2.filter2D(img, -1, sharpening_mask1)
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


    apt1 = region_of_interest1_up(img, x, y, radius)
    apt2 = region_of_interest2_up(img, x, y, radius)
    apt3 = region_of_interest3_up(img, x, y, radius)
    apt4 = region_of_interest4_up(img, x, y, radius)
    apt5 = region_of_interest1_down(img, x, y, radius)
    apt6 = region_of_interest2_down(img, x, y, radius)
    apt7 = region_of_interest3_down(img, x, y, radius)
    apt8 = region_of_interest4_down(img, x, y, radius)
    apt9 = region_of_interest5_up(img, x, y, radius)
    apt10 = region_of_interest6_up(img, x, y, radius)
    apt11 = region_of_interest7_up(img, x, y, radius)
    apt12 = region_of_interest8_up(img, x, y, radius)
    apt13 = region_of_interest5_down(img, x, y, radius)
    apt14 = region_of_interest6_down(img, x, y, radius)
    apt15 = region_of_interest7_down(img, x, y, radius)
    apt16 = region_of_interest8_down(img, x, y, radius)

    '''
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
    '''
    apt1 = adapt_thres(apt1)
    apt2 = adapt_thres(apt2)
    apt3 = adapt_thres(apt3)
    apt4 = adapt_thres(apt4)
    apt5 = adapt_thres(apt5)
    apt6 = adapt_thres(apt6)
    apt7 = adapt_thres(apt7)
    apt8 = adapt_thres(apt8)
    apt9 = adapt_thres(apt9)
    apt10 = adapt_thres(apt10)
    apt11 = adapt_thres(apt11)
    apt12 = adapt_thres(apt12)
    apt13 = adapt_thres(apt13)
    apt14 = adapt_thres(apt14)
    apt15 = adapt_thres(apt15)
    apt16 = adapt_thres(apt16)

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



import numpy as np
import matplotlib.pyplot as plt

img = np.fromfile("90.ARW", dtype=np.uint32)
print(img.size) #check your image size, say 1048576
#shape it accordingly, that is, 1048576=1024*1024
img.shape = (1516,4112)

plt.imshow(img)
plt.savefig("yourNewImage.png")



#gray = grayscale(img)

        # 원 영역 검출
kernel_size = 391
#blur_gray = gaussian_blur(gray, kernel_size)
#apt = thresholding(blur_gray)
#x, y,radius = contour(apt)
#print(x, y, radius)
#m=masking(gray,x,y,radius)
plt.figure(figsize=(10, 8))
plt.imshow(img, cmap='gray')
plt.show()




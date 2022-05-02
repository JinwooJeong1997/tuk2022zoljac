import math
import cv2
import numpy as np
import rawpy
import imageio
import matplotlib.pyplot as plt
def raw_to_jpg(name):
    try :
        raw = rawpy.imread(name)
        
        
        im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=False, output_bps=16)
        rgb = np.float32(im / 65535.0 * 255.0)
      
    #RAW이외 이미지 처리용 루틴  
    except Exception as e:
        print(e)
        raw = cv2.imread(name,cv2.IMREAD_UNCHANGED)
        dst = remove_shadow(raw)
        dst = cv2.resize(dst,dsize=(6024,4024),interpolation=cv2.INTER_LINEAR)
        
        #dst = cv2.fastNlMeansDenoisingColored(dst,None,7,7,7,21)
        imageio.imsave('detect/sample_denoise.jpg', dst)
        
        dst = cv2.normalize(dst,None,0,255,cv2.NORM_MINMAX)
        dst = contrast(dst)
        k = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        dst = cv2.morphologyEx(dst,cv2.MORPH_OPEN,k)
        dst = cv2.medianBlur(dst,3)
        dst = cv2.bilateralFilter(dst,-1,10,5)
        dst = cv2.bitwise_not(dst)
        #dst = contrast(dst)
        
        
        #detect_circle(dst)
        im = np.array(dst)
        rgb = np.float32(im)
    print(type(im))
    print(im)
    h,w,c = rgb.shape
    print("{} {} {}".format(h,w,c))
    rgb = np.asarray(rgb, np.uint8)
    imageio.imsave('detect/sample.jpg', rgb)
    return rgb

#원 검출
def detect_circle(img):
    cir_img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,param1=50,param2=25,minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        cv2.circle(cir_img,(i[0],i[1]),i[2],(0,255,0),2)
    imageio.imsave('detect/sample_circle.jpg',cir_img)

#그림자 제거
def remove_shadow(img):
    rgb_planes = cv2.split(img)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    return result

#대비조절 함수
def contrast(img):
    lab = cv2.cvtColor(img,cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg,cv2.COLOR_LAB2BGR)
    imageio.imsave('detect/sample_contrast.jpg', final)
    return final

def grayscale(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imageio.imsave('detect/sample_gray.jpg', gray)
    return gray

def gaussian_blur(img, kernel_size):
    img_gb = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    imageio.imsave('detect/sample_gaussian.jpg', img_gb)
    return img_gb

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

def region_of_interest9_up(img,x,y,radius):
    square=int(0.13*radius)
    masked_image=img[int(y)-5*square:int(y)-4*square,int(x)+2*square:int(x)+3*square].copy()
    return masked_image

def region_of_interest10_up(img,x,y,radius):
    square = int(0.13 * radius)
    masked_image = img[int(y) - 5 * square:int(y) - 4 * square, int(x) + 1 * square:int(x) + 2 * square].copy()
    return masked_image

def region_of_interest11_up(img,x,y,radius):
    square=int(0.13*radius)
    masked_image=img[int(y)-4*square:int(y)-3*square,int(x)+2*square:int(x)+3*square].copy()
    return masked_image

def region_of_interest12_up(img,x,y,radius):
    square = int(0.13 * radius)
    masked_image = img[int(y) - 4 * square:int(y) - 3 * square, int(x) + 1 * square:int(x) + 2 * square].copy()
    return masked_image

def region_of_interest9_down(img,x,y,radius):
    square=int(0.13*radius)
    masked_image=img[int(y)+4*square:int(y)+5*square,int(x)+2*square:int(x)+3*square].copy()
    return masked_image

def region_of_interest10_down(img,x,y,radius):
    square = int(0.13 * radius)
    masked_image = img[int(y) +4 * square:int(y) +5 * square, int(x) + 1 * square:int(x) + 2 * square].copy()
    return masked_image

def region_of_interest11_down(img,x,y,radius):
    square=int(0.13*radius)
    masked_image=img[int(y)+3*square:int(y)+4*square,int(x)+2*square:int(x)+3*square].copy()
    return masked_image

def region_of_interest12_down(img,x,y,radius):
    square = int(0.13 * radius)
    masked_image = img[int(y) +3 * square:int(y) +4 * square, int(x) + 1 * square:int(x) + 2 * square].copy()
    return masked_image

def region_of_interest13_up(img,x,y,radius):
    square=int(0.13*radius)
    masked_image=img[int(y)-5*square:int(y)-4*square,int(x)-3*square:int(x)-2*square].copy()
    return masked_image

def region_of_interest14_up(img,x,y,radius):
    square = int(0.13 * radius)
    masked_image = img[int(y) - 5 * square:int(y) - 4 * square, int(x) + -2 * square:int(x) -1 * square].copy()
    return masked_image

def region_of_interest15_up(img,x,y,radius):
    square=int(0.13*radius)
    masked_image=img[int(y)-4*square:int(y)-3*square,int(x)-3*square:int(x)-2*square].copy()
    return masked_image

def region_of_interest16_up(img,x,y,radius):
    square = int(0.13 * radius)
    masked_image = img[int(y) - 4 * square:int(y) - 3 * square, int(x) -2 * square:int(x) -1 * square].copy()
    return masked_image

def region_of_interest13_down(img,x,y,radius):
    square=int(0.13*radius)
    masked_image=img[int(y)+4*square:int(y)+5*square,int(x)-3*square:int(x)-2*square].copy()
    return masked_image

def region_of_interest14_down(img,x,y,radius):
    square = int(0.13 * radius)
    masked_image = img[int(y) +4 * square:int(y) +5 * square, int(x) -2 * square:int(x) -1 * square].copy()
    return masked_image

def region_of_interest15_down(img,x,y,radius):
    square=int(0.13*radius)
    masked_image=img[int(y)+3*square:int(y)+4*square,int(x)-3*square:int(x)-2*square].copy()
    return masked_image

def region_of_interest16_down(img,x,y,radius):
    square = int(0.13 * radius)
    masked_image = img[int(y) +3 * square:int(y) +4 * square, int(x) -2 * square:int(x) -1 * square].copy()
    return masked_image

def region_of_interest_mamo(x,p):
    img=cv2.imread(p+'/rotate.jpg')
    square =150
    print(square)
    for i in range(1,10):
        print((2*i)*square, (2*i+2)*square, int(x)-1 * square,int(x)+1*square)
        a=img[(2*i)*square:(2*i+2)*square, int(x)-1 * square:int(x)+1*square]
        cv2.imwrite(p+'/mamo/'+str(i)+'.jpg',a)

def region_of_interest_make(img,x,y):
    masked_image = img[int(y)-150:int(y)+150, int(x)-150:int(x)+150 ].copy()
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

def region_of_interest_white(img,x,y,radius):
    mask=np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,)
    else:
        ignore_mask_color = 255

    cv2.circle(mask, (int(x),int(y)), int(radius), (255, 255, 255), -1)
    mask1=cv2.bitwise_not(mask)
    masked_image=cv2.bitwise_or(img,mask1)
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
        line_img='None'
    return line_img, a

def num_hough_lines(img,rho,theta,threshold,min_line_len,max_line_gap):
    lines=cv2.HoughLinesP(img,rho,theta,threshold,np.array([]),
                          minLineLength=min_line_len,
                          maxLineGap=max_line_gap)
    if lines is not None:
        b=len(lines)
    else:
        b = 0
    return b

def hough_circle(img):
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 5, 3000, None, 20,500,3000)
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

def rotate(img,val):
    height, width = img.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), -int(val), 1)
    dst = cv2.warpAffine(img, matrix, (width, height))
    return dst

def wrap(img,matrix):
    height, width = img.shape
    dst = cv2.warpAffine(img, matrix, (width, height))
    return dst

def thresholding(img):
    ret,thr=cv2.threshold(img,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    imageio.imsave('detect/sample_thr.jpg', thr)
    return thr

def thresholding_1(img):
    ret,thr=cv2.threshold(img,100,255,cv2.THRESH_BINARY)
    imageio.imsave('detect/sample_thr.jpg', thr)
    return thr

def contour(img):
    contours,_=cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    contour=contours[0]
    (x, y), radius = cv2.minEnclosingCircle(contour)
    
    tImg = cv2.drawContours(img, contours, -1, (0,255,0), 3)
    imageio.imsave('detect/sample_contour.jpg', tImg)
    return x,y,radius

def sharpening(img):
    sharpening_mask = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpening = cv2.filter2D(img, -1, sharpening_mask)
    return sharpening

def calc4(a,b,c,d,a1,b1,c1,d1):
    arr=[a,b,c,d,a1,b1,c1,d1]

    flag=1
    for i in arr:
        if i == 'None':
            flag=0
    if flag==0:
        return None
    elif flag==1:
        ab_arr1=[abs(a),abs(b),abs(c),abs(d),abs(a1),abs(b1),abs(c1),abs(d1)]
        arr1=[a,b,c,d,a1,b1,c1,d1]
        new_arr=[]
        arr_a=[]
        arr_b=[]
        ab_sum1=0
        ab_cnt1=0
        cnt90_1=0
        cnt0_1=0
        sum1=0
        cnt1=0
        for i in ab_arr1:
            if i !='None':
                ab_sum1+=i
                ab_cnt1+=1

        ab=ab_sum1/ab_cnt1

        if ab>=83 and abs(sum(ab_arr1)-sum(arr1))>100: avg1=ab
        else:
            for i in arr1:
                if i !='None' and i!=90 and i!=0:
                    sum1+=i
                    cnt1+=1
                    new_arr.append(i)
                elif i==90:
                    cnt90_1+=1
                elif i==0:
                    cnt0_1+=1

            for i in new_arr:
                if abs(sum(new_arr)/len(new_arr)-i)>20:
                    new_arr.remove(i)

            sum1=sum(new_arr)
            cnt1=len(new_arr)

            if sum1/cnt1 >= 85:
                avg1 = (sum1 + 90 * cnt90_1) / (cnt1 + cnt90_1)
            elif sum1/cnt1 >= 5:
                avg1= sum1 / cnt1
            elif sum1/cnt1 >= -5:
                avg1= sum1 / (cnt1+cnt0_1)
            elif sum1/cnt1 >= -85:
                avg1= sum1/cnt1
            elif sum1/cnt1 >= -90:
                avg1= (sum1 - 90 * cnt90_1) / (cnt1 + cnt90_1)
            else: avg1='None'

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
    mask9_up = region_of_interest9_up(img, x, y, radius)
    mask10_up = region_of_interest10_up(img, x, y, radius)
    mask11_up = region_of_interest11_up(img, x, y, radius)
    mask12_up = region_of_interest12_up(img, x, y, radius)
    mask9_d = region_of_interest9_down(img, x, y, radius)
    mask10_d = region_of_interest10_down(img, x, y, radius)
    mask11_d = region_of_interest11_down(img, x, y, radius)
    mask12_d = region_of_interest12_down(img, x, y, radius)
    mask13_up = region_of_interest13_up(img, x, y, radius)
    mask14_up = region_of_interest14_up(img, x, y, radius)
    mask15_up = region_of_interest15_up(img, x, y, radius)
    mask16_up = region_of_interest16_up(img, x, y, radius)
    mask13_d = region_of_interest13_down(img, x, y, radius)
    mask14_d = region_of_interest14_down(img, x, y, radius)
    mask15_d = region_of_interest15_down(img, x, y, radius)
    mask16_d = region_of_interest16_down(img, x, y, radius)

    '''
    imageio.imsave('detect/1u.jpg',mask1_up)
    imageio.imsave('detect/2u.jpg',mask2_up)
    imageio.imsave('detect/3u.jpg',mask3_up)
    imageio.imsave('detect/4u.jpg',mask4_up)
    imageio.imsave('detect/5u.jpg',mask5_up)
    imageio.imsave('detect/6u.jpg',mask6_up)
    imageio.imsave('detect/7u.jpg',mask7_up)
    imageio.imsave('detect/8u.jpg',mask8_up)
    imageio.imsave('detect/9u.jpg',mask9_up)
    imageio.imsave('detect/10u.jpg',mask10_up)
    imageio.imsave('detect/11u.jpg',mask11_up)
    imageio.imsave('detect/12u.jpg',mask12_up)
    imageio.imsave('detect/13u.jpg',mask13_up)
    imageio.imsave('detect/14u.jpg',mask14_up)
    imageio.imsave('detect/15u.jpg',mask15_up)
    imageio.imsave('detect/16u.jpg',mask16_up)
    
    imageio.imsave('detect/1d.jpg',mask1_d)
    imageio.imsave('detect/2d.jpg',mask2_d)
    imageio.imsave('detect/3d.jpg',mask3_d)
    imageio.imsave('detect/4d.jpg',mask4_d)
    imageio.imsave('detect/5d.jpg',mask5_d)
    imageio.imsave('detect/6d.jpg',mask6_d)
    imageio.imsave('detect/7d.jpg',mask7_d)
    imageio.imsave('detect/8d.jpg',mask8_d)
    imageio.imsave('detect/9d.jpg',mask9_d)
    imageio.imsave('detect/10d.jpg',mask10_d)
    imageio.imsave('detect/11d.jpg',mask11_d)
    imageio.imsave('detect/12d.jpg',mask12_d)
    imageio.imsave('detect/13d.jpg',mask13_d)
    imageio.imsave('detect/14d.jpg',mask14_d)
    imageio.imsave('detect/15d.jpg',mask15_d)
    imageio.imsave('detect/16d.jpg',mask16_d)
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
    mask9_up = sharpening(mask9_up)
    mask10_up = sharpening(mask10_up)
    mask11_up = sharpening(mask11_up)
    mask12_up = sharpening(mask12_up)
    mask9_d = sharpening(mask9_d)
    mask10_d = sharpening(mask10_d)
    mask11_d = sharpening(mask11_d)
    mask12_d = sharpening(mask12_d)
    mask13_up = sharpening(mask13_up)
    mask14_up = sharpening(mask14_up)
    mask15_up = sharpening(mask15_up)
    mask16_up = sharpening(mask16_up)
    mask13_d = sharpening(mask13_d)
    mask14_d = sharpening(mask14_d)
    mask15_d = sharpening(mask15_d)
    mask16_d = sharpening(mask16_d)

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
    apt17 = adapt_thres(mask9_up)
    apt18 = adapt_thres(mask10_up)
    apt19 = adapt_thres(mask11_up)
    apt20 = adapt_thres(mask12_up)
    apt21 = adapt_thres(mask9_d)
    apt22 = adapt_thres(mask10_d)
    apt23 = adapt_thres(mask11_d)
    apt24 = adapt_thres(mask12_d)
    apt25 = adapt_thres(mask13_up)
    apt26 = adapt_thres(mask14_up)
    apt27 = adapt_thres(mask15_up)
    apt28 = adapt_thres(mask16_up)
    apt29 = adapt_thres(mask13_d)
    apt30 = adapt_thres(mask14_d)
    apt31 = adapt_thres(mask15_d)
    apt32 = adapt_thres(mask16_d)

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
    lines17, a1 = hough_lines(apt17, rho, theta, threshold, min_line_len, max_line_gap)
    lines18, b1 = hough_lines(apt18, rho, theta, threshold, min_line_len, max_line_gap)
    lines19, c1 = hough_lines(apt19, rho, theta, threshold, min_line_len, max_line_gap)
    lines20, d1 = hough_lines(apt20, rho, theta, threshold, min_line_len, max_line_gap)
    lines21, e1 = hough_lines(apt21, rho, theta, threshold, min_line_len, max_line_gap)
    lines22, f1 = hough_lines(apt22, rho, theta, threshold, min_line_len, max_line_gap)
    lines23, g1 = hough_lines(apt23, rho, theta, threshold, min_line_len, max_line_gap)
    lines24, h1 = hough_lines(apt24, rho, theta, threshold, min_line_len, max_line_gap)
    lines25, i1 = hough_lines(apt25, rho, theta, threshold, min_line_len, max_line_gap)
    lines26, j1 = hough_lines(apt26, rho, theta, threshold, min_line_len, max_line_gap)
    lines27, k1 = hough_lines(apt27, rho, theta, threshold, min_line_len, max_line_gap)
    lines28, l1 = hough_lines(apt28, rho, theta, threshold, min_line_len, max_line_gap)
    lines29, m1 = hough_lines(apt29, rho, theta, threshold, min_line_len, max_line_gap)
    lines30, n1 = hough_lines(apt30, rho, theta, threshold, min_line_len, max_line_gap)
    lines31, o1 = hough_lines(apt31, rho, theta, threshold, min_line_len, max_line_gap)
    lines32, p1 = hough_lines(apt32, rho, theta, threshold, min_line_len, max_line_gap)

    sum_A = calc4(a, b, c, d,a1,b1,c1,d1)
    sum_B = calc4(e, f, g, h,e1, f1, g1, h1)
    sum_C = calc4(i, j, k, l,i1, j1, k1, l1)
    sum_D = calc4(m, n, o, p,m1, n1, o1, p1)
    print(a, b, c, d, a1, b1, c1, d1, sum_A)
    print(e, f, g, h, e1, f1, g1, h1, sum_B)
    print(i, j, k, l,i1, j1, k1, l1, sum_C)
    print(m, n, o, p, m1, n1, o1, p1, sum_D)


    return sum_A,sum_B,sum_C,sum_D

def num_masking(img,x,y,radius):


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
    apt17 = region_of_interest9_up(img, x, y, radius)
    apt18 = region_of_interest10_up(img, x, y, radius)
    apt19 = region_of_interest11_up(img, x, y, radius)
    apt20 = region_of_interest12_up(img, x, y, radius)
    apt21 = region_of_interest9_down(img, x, y, radius)
    apt22 = region_of_interest10_down(img, x, y, radius)
    apt23 = region_of_interest11_down(img, x, y, radius)
    apt24 = region_of_interest12_down(img, x, y, radius)
    apt25 = region_of_interest13_up(img, x, y, radius)
    apt26 = region_of_interest14_up(img, x, y, radius)
    apt27 = region_of_interest15_up(img, x, y, radius)
    apt28 = region_of_interest16_up(img, x, y, radius)
    apt29 = region_of_interest13_down(img, x, y, radius)
    apt30 = region_of_interest14_down(img, x, y, radius)
    apt31 = region_of_interest15_down(img, x, y, radius)
    apt32 = region_of_interest16_down(img, x, y, radius)
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
    mask9_up = sharpening(mask9_up)
    mask10_up = sharpening(mask10_up)
    mask11_up = sharpening(mask11_up)
    mask12_up = sharpening(mask12_up)
    mask9_d = sharpening(mask9_d)
    mask10_d = sharpening(mask10_d)
    mask11_d = sharpening(mask11_d)
    mask12_d = sharpening(mask12_d)
    mask13_up = sharpening(mask13_up)
    mask14_up = sharpening(mask14_up)
    mask15_up = sharpening(mask15_up)
    mask16_up = sharpening(mask16_up)
    mask13_d = sharpening(mask13_d)
    mask14_d = sharpening(mask14_d)
    mask15_d = sharpening(mask15_d)
    mask16_d = sharpening(mask16_d)
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
    apt17 = adapt_thres(apt17)
    apt18 = adapt_thres(apt18)
    apt19 = adapt_thres(apt19)
    apt20 = adapt_thres(apt20)
    apt21 = adapt_thres(apt21)
    apt22 = adapt_thres(apt22)
    apt23 = adapt_thres(apt23)
    apt24 = adapt_thres(apt24)
    apt25 = adapt_thres(apt25)
    apt26 = adapt_thres(apt26)
    apt27 = adapt_thres(apt27)
    apt28 = adapt_thres(apt28)
    apt29 = adapt_thres(apt29)
    apt30 = adapt_thres(apt30)
    apt31 = adapt_thres(apt31)
    apt32 = adapt_thres(apt32)
    # 라인검출

    rho = 2
    theta = np.pi / 180
    threshold = 300
    min_line_len = 100
    max_line_gap = 150
    line=[]
    line.append(num_hough_lines(apt1, rho, theta, threshold, min_line_len, max_line_gap))
    line.append(num_hough_lines(apt2, rho, theta, threshold, min_line_len, max_line_gap))
    line.append(num_hough_lines(apt3, rho, theta, threshold, min_line_len, max_line_gap))
    line.append(num_hough_lines(apt4, rho, theta, threshold, min_line_len, max_line_gap))
    line.append(num_hough_lines(apt5, rho, theta, threshold, min_line_len, max_line_gap))
    line.append(num_hough_lines(apt6, rho, theta, threshold, min_line_len, max_line_gap))
    line.append(num_hough_lines(apt7, rho, theta, threshold, min_line_len, max_line_gap))
    line.append(num_hough_lines(apt8, rho, theta, threshold, min_line_len, max_line_gap))
    line.append(num_hough_lines(apt9, rho, theta, threshold, min_line_len, max_line_gap))
    line.append(num_hough_lines(apt10, rho, theta, threshold, min_line_len, max_line_gap))
    line.append(num_hough_lines(apt11, rho, theta, threshold, min_line_len, max_line_gap))
    line.append(num_hough_lines(apt12, rho, theta, threshold, min_line_len, max_line_gap))
    line.append(num_hough_lines(apt13, rho, theta, threshold, min_line_len, max_line_gap))
    line.append(num_hough_lines(apt14, rho, theta, threshold, min_line_len, max_line_gap))
    line.append(num_hough_lines(apt15, rho, theta, threshold, min_line_len, max_line_gap))
    line.append(num_hough_lines(apt16, rho, theta, threshold, min_line_len, max_line_gap))
    line.append(num_hough_lines(apt17, rho, theta, threshold, min_line_len, max_line_gap))
    line.append(num_hough_lines(apt18, rho, theta, threshold, min_line_len, max_line_gap))
    line.append(num_hough_lines(apt19, rho, theta, threshold, min_line_len, max_line_gap))
    line.append(num_hough_lines(apt20, rho, theta, threshold, min_line_len, max_line_gap))
    line.append(num_hough_lines(apt21, rho, theta, threshold, min_line_len, max_line_gap))
    line.append(num_hough_lines(apt22, rho, theta, threshold, min_line_len, max_line_gap))
    line.append(num_hough_lines(apt23, rho, theta, threshold, min_line_len, max_line_gap))
    line.append(num_hough_lines(apt24, rho, theta, threshold, min_line_len, max_line_gap))
    line.append(num_hough_lines(apt25, rho, theta, threshold, min_line_len, max_line_gap))
    line.append(num_hough_lines(apt26, rho, theta, threshold, min_line_len, max_line_gap))
    line.append(num_hough_lines(apt27, rho, theta, threshold, min_line_len, max_line_gap))
    line.append(num_hough_lines(apt28, rho, theta, threshold, min_line_len, max_line_gap))
    line.append(num_hough_lines(apt29, rho, theta, threshold, min_line_len, max_line_gap))
    line.append(num_hough_lines(apt30, rho, theta, threshold, min_line_len, max_line_gap))
    line.append(num_hough_lines(apt31, rho, theta, threshold, min_line_len, max_line_gap))
    line.append(num_hough_lines(apt32, rho, theta, threshold, min_line_len, max_line_gap))

    ma=max(line)
    mi=min(line)
    av=sum(line)/len(line)


    return ma,mi,av

def basic_showImg(img):

    plt.figure(figsize=(10,9))
    plt.imshow(img,cmap='gray')
    plt.show()

def image_change(loc,x,p):
    img=cv2.imread(p+'/rotate.jpg')
    for i in range(len(loc)):
        a=cv2.imread(p+'/matter/exp/'+str(i)+'.jpg')
        img[int(loc[i][1]) - 150:int(loc[i][1]) + 150, int(loc[i][0]) - 150:int(loc[i][0]) + 150]=a
    for i in range(1,10):
        square=150
        a=cv2.imread(p+'/mamo/exp/'+str(i)+'.jpg')
        img[(2*i)*square:(2*i+2)*square, int(x)-1 * square:int(x)+1*square]=a

    cv2.imwrite(p+'/matter.jpg', img)

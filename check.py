import header as h
import cv2
import imageio

def circle_analy( name):
 # 시작 시간 저장

    # 작업 코드

    img_name = name
    img = h.raw_to_jpg(img_name)
    gray = h.grayscale(img)

    # 원 영역 검출
    kernel_size = 331
    blur_gray = h.gaussian_blur(gray, kernel_size)
    apt = h.thresholding(blur_gray)

    x, y, radius = h.contour(apt)
    print(x, y, radius)

    # 마스크
    sum_A, sum_B, sum_C, sum_D = h.masking(gray, x, y, radius)
    rotate = 0
    if sum_A is None or sum_B is None or sum_C is None or sum_D is None:
        rotate = '9999999'
    elif int(sum_A) * int(sum_B) < 0 or abs(abs(sum_A) - abs(sum_B)) <= 3 or abs(sum_A) > 85 or abs(sum_B) > 85:
        height, width = gray.shape
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 30, 1)
        dst = cv2.warpAffine(gray, matrix, (width, height))
        sum_A, sum_B, sum_C, sum_D = h.masking(dst, x, y, radius)
        avg = (sum_A + sum_B + sum_C + sum_D) / 4
        if abs(sum_B) + abs(sum_D) >= abs(sum_A) + abs(sum_C):
            rotate = avg - 30
        else:
            rotate = 180 + avg - 30

    else:
        avg = (sum_A + sum_B + sum_C + sum_D) / 4
        if abs(sum_B) + abs(sum_D) >= abs(sum_A) + abs(sum_C):
            rotate = avg
        else:
            rotate = 180 + avg



img=cv2.imread('img/n0.png')
gray = h.grayscale(img)

# 원 영역 검출
kernel_size = 331
blur_gray = h.gaussian_blur(gray, kernel_size)
apt = h.thresholding(blur_gray)

x, y, radius = h.contour(apt)
print(x, y, radius)
h.basic_showImg(apt)

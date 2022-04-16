import cv2
from pypylon_opencv_viewer import BaslerOpenCVViewer
from pypylon import pylon
import matplotlib.pyplot as plt
import configparser

#카메라 연결 관련

def USB_camera():
    port=0
    flag = 1
    cap = cv2.VideoCapture(port)
    print(cap)
    if cap.get(3)!=0:

        print('width :%d, height : %d' % (cap.get(3), cap.get(4)))

        while(True):
            ret, frame = cap.read()    # Read 결과와 frame

            if(ret) :
                gray = cv2.cvtColor(frame,  cv2.COLOR_BGR2GRAY)    # 입력 받은 화면 Gray로 변환

                cv2.imshow('frame_color', frame)    # 컬러 화면 출력
                cv2.imshow('frame_gray', gray)    # Gray 화면 출력
                if cv2.waitKey(1) == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()

    else: flag=0

    return flag

def ETHERNET_camera():
    camera = configparser.ConfigParser()
    camera.read('camera.ini', encoding='utf-8')
    flag=1
    IPNum = camera['SET']['var2']

    info = None

    for i in pylon.TlFactory.GetInstance().EnumerateDevices():
        if i.GetDeviceClass() == 'BaslerGigE':
            if i.GetIpAddress() == IPNum:
                info = i
            break
    else:
        print('Camera with IP {} not found'.format(IPNum))
        flag=0

    # VERY IMPORTANT STEP! To use Basler PyPylon OpenCV viewer you have to call .Open() method on you camera
    if info is not None:
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(info))
        camera.Open()

        viewer = BaslerOpenCVViewer(camera)
        img=viewer._run_continuous_shot()


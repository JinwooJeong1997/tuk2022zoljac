#웹캠 읽어오기
import cv2
import datetime

class camera_remote:
    def __init__(self) :
        video_capture = cv2.VideoCapture(0)
        
    def check_cam(self) :
        self.capture()
        
    def capture(self):
        grapped, frame = self.video_capture.read()
        
        file = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")+".jpg"
        cv2.imwrite(file,frame)
        
    def __del__(self):
        self.video_capture.release()
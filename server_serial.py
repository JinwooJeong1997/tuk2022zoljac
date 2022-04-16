
# 현 상황
# 1. xgt 프로토콜 메뉴얼을 통해 읽기/쓰기 구현
# 해야할 일
# 1. plc 통신이 되는지 확인
# 2. 통신이 가능하면 데이터값을 읽어 보내는 기능 구현하기

from dataclasses import dataclass, field
import serial
import threading

# 통신용 클래스
@dataclass(unsafe_hash=True)
class ReadCommd:
    Station_no : str
    Instruction : str
    No_of_blocks : str
    Variable_Length : str
    Variable_Name : str
    
    @property
    def protocol(self):
        cmd = '\x05{0}{1}{2}{3}{4}\x04'.format(
            self.Station_no,
            self.Instruction,
            self.No_of_blocks,
            self.Variable_Length,
            self.Variable_Name)
        #문자열-->바이트
        return cmd.encode(encoding="utf-8")

class server_serial:
    #시리얼 포트 연결(COM1,9600,n,1,8) 
    def __init__(self):
       self.ser = serial.Serial(
            port='COM1', 
            baudrate = 9600,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            timeout=1
            )
    
    #주기별 쓰레드 시작(초)
    def start_thread(self,period) :
        threading.Timer(period,self.readData).start
    
    # 읽기 명령 생성
    def readCmd(self):
        self.tx = ReadCommd('00','RSS','01','06','%MW100').protocol
    
    # 데이터 쓰기  
    def writeData(self):
        self.ser.write(self.tx, encoding = "utf-8")
 
    #데이터 읽기
    def readData(self):
        rx = self.ser.readline().decode('utf-8')
        if rx[:1] == chr(6):    # <ACK> (0x06)
            int(rx[-5:-1], 16)   # 16진수로 리턴되는 카운트를 10진수로 변환
        elif rx[:1] == chr(21): # <NAK> (0x15)
            print('Error = ' + rx[-5:-1])  # 부록 3.1 XGT 서버 에러코드'를 참고
          
    def __del__(self):
        # 포트닫기
        self.ser.close()
    
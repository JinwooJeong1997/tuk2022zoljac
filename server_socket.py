#socket test
#
# 1. D1000 번지 값 read request (per 1sec) - read check
# 1-1. if ok, do capture
# 1-2. if not, continue
# 2. D1000 write request (write 0-false-)
# 3. D1004 read request  - read 4 words (MM DD HH mm) -> mkdir "MMDDHHmm"
# 4. D1002 write request ( 1.ng 2.ok)

# packet = header + frame
# header
#   - comp id 
#   - reserve (2)
#   - PLC 정보
#   - CPU
#   - H33 (1)
#   - invokeID(2) - checksum for pc hmi error check , can set randomly
#   - length (2) - frame length
#   - reserve (1)
#   - BCC

# request = header + inst + type + reserve(2) + data
# ack = header + inst + type + reserve(2) + error(2)"h0000" + data
# Nak = header + inst + type + reserve(2) + error(2)"not h0000" + errorcode(2)

from socket import *
from select import *
import sys
from time import ctime
import asyncio 

import binascii

HOST = '192.168.100.10'
PORT = 2004
BUFSIZE = 4096
ADDR = (HOST, PORT)

class xgt_header:
    compID = "4C5349532D58474C" # LSIS-XGL
    reserve = "0000" #fixed
    plcInfo = "0000" #fiexd
    cpuInfo = "A0" #xgk-a0 xgi-a4 xgr-a8
    sourceOfFrame = "33" # cli->ser : 0x33 ser->cli : 0x11
    invokeId = "0001" 
    length = "1100"
    FnetPos = "03"
    bcc = "45" #header byte sum

    def getHeader(self):
        return self.compID+self.reserve+self.plcInfo+self.cpuInfo+self.sourceOfFrame+self.invokeId+self.length+self.FnetPos+self.bcc

    #def setlength(self,data) :
    #    length = hex(len(hex(data))-2)

    #def setBCC(self) : #헤더값을 전부 더한 값의 2자리 값
    #    bcc 

    

#프레임 작성 시 프레임에서 
#16 진수 워드 데이터를 표현할 때는 숫자 앞의 h 를 빼고, 
#두 바이트의 위치를 바꾸어 주어야 합니다.
#예) h0054 ⇒5400

class req_packet :
    inst = "5400" #read req
    dtype = "0200"
    reserve = "0000"
    # 블록수  : [변수길이][변수] h0001~h0010
    # 변수 길이(변수 이름 길이) h01~h10
    # datapos  : %(워드가능한 영역)W(위치)
    datamount = "0100" # h0001
    datalength = "0700" 
    datapos = "25 44 57 31 30 30 30" # %DW1000
    
    def getPacket(self):
        return self.inst+self.dtype+self.reserve+self.datamount+self.datalength+self.datapos


xgt_hd = xgt_header()
xgt_req = req_packet()
print("{}".format(xgt_hd.getHeader()))
print("{}".format(xgt_req.getPacket()))
print(("LSIS-XGT").encode('utf-8').hex())
print(("%MW1000").encode('utf-8').hex())
print(("%MW1000").encode('utf-8').hex())

clientSocket = socket(AF_INET, SOCK_STREAM)  # 서버에 접속하기 위한 소켓을 생성한다.
def connect():  
    try:
        clientSocket.connect(ADDR)  # 서버에 접속을 시도한다.

    except  Exception as e:
        print('%s:%s' % ADDR)
        sys.exit()
    print('connect is success')

def datarecv():
    #sendData = b'4C5349532D58475400000000A0330200110003405400020000000100070025445731303030'
    xgt = xgt_hd.getHeader()+xgt_req.getPacket()                             
    print(xgt)    
    sendData =  b"4C5349532D58475400000000A0330000110003005400020000000100040025445731303031"
    print("send")
    #print(type(sendData))
    #print(sendData)
    #print(sendData.__sizeof__())
    #print(binascii.hexlify(sendData).decode("ascii"))
    clientSocket.send(sendData)
    recvData = clientSocket.recv(6000)
    print(type(recvData))
    print("recv")
    print(binascii.hexlify(recvData))
    print(recvData.__sizeof__())




connect()
datarecv()

clientSocket.close()
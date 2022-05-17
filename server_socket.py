#socket test
#
# 1. D1000 번지 값 read request (per 1sec) - read check
# 1-1. if ok, do capture
# 1-2. if not, continue
# 2. D1000 write request (write 0-false-)
# 3. D1004 read request  - read 4 words (MM DD HH mm) -> mkdir "MMDDHHmm"
# 4. D1002 write request ( 1.ng 2.ok)
# 16byte word -> swap ex) h 0012 -> 1200

# packet = header + frame
# header
#   - comp id 
#   - reserve (2)
#   - PLC 정보
#   - CPU
#   - H33 (1)
#   - invokeID(2) - checksum for pc hmi error check , can set randomly
#   - length (2) - body length
#   - reserve (1)
#   - BCC - sum of header hex

# request = header + inst + type + reserve(2) + data
# ack = header + inst + type + reserve(2) + error(2)"h0000" + data
# Nak = header + inst + type + reserve(2) + error(2)"not h0000" + errorcode(2)

from base64 import decode
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
    compID = 0x4C5349532D58474C # LSIS-XGL
    reserve = 0x0000 #fixed
    plcInfo = 0x0000 #fiexd
    cpuInfo = 0xA0 #xgk-a0 xgi-a4 xgr-a8
    sourceOfFrame = 0x33 # cli->ser : 0x33 ser->cli : 0x11
    invokeId = 0x0001 
    length = 0x1100
    FnetPos = 0x03
    bcc = 0x45 #header byte sum
    
    def get_packet(self):
        buf = self.compID
        #print("{}".format(hex(buf)))
        buf = (buf * pow(16,4)) + self.reserve
        #print("{}".format(hex(buf)))
        buf = (buf * pow(16,4)) + self.plcInfo
        #print("{}".format(hex(buf)))
        buf = (buf * pow(16,2)) + self.cpuInfo
        #print("{}".format(hex(buf)))
        buf = (buf * pow(16,2)) + self.sourceOfFrame
        #print("{}".format(hex(buf)))
        buf = (buf * pow(16,4)) + self.invokeId
        #print("{}".format(hex(buf)))
        buf = (buf * pow(16,4)) + self.length
        #print("{}".format(hex(buf)))
        buf = (buf * pow(16,2)) + self.FnetPos
        #print("{}".format(hex(buf)))
        buf = (buf * pow(16,2)) + self.bcc
        #print("{}".format(hex(buf)))
        return buf
    
    #def set_length(self):
    
    def set_bcc(self): # BCC 계산 (필드 변경시 호출할것)
        hexs =[hex(self.get_packet())[i:i+2] for i in range(2,len(hex(self.get_packet()))-2,2)]
        print(hexs)
        result = 0
        for i in hexs :
            #print("getbcc: {} + {}".format(hex(result),hex(int(i,16))))
            interger = int(i,16)
            result += interger
        print(hex(result)[len(hex(result))-2:])
        self.bcc = int(hex(result)[len(hex(result))-2:],16)
    
        
    

    
#프레임 작성 시 프레임에서 
#16 진수 워드 데이터를 표현할 때는 숫자 앞의 h 를 빼고, 
#두 바이트의 위치를 바꾸어 주어야 합니다.
#예) h0054 ⇒5400
class xgt_body :
    # 블록수  : [변수길이][변수] h0001~h0010
    # 변수 길이(변수 이름 길이) h01~h10
    # datapos  : %(워드가능한 영역)W(위치)
    inst = 0x5400 #read req
    dtype = 0x0200
    reserve = 0x0000
    datamount = 0x0100 # h0001
    datalength = 0x0700 
    datapos = 0x25445731303030 # %DW1000
    
    def get_packet(self):
        buf = self.inst
        #print("{}".format(hex(buf)))
        buf = (buf * pow(16,4)) + self.dtype
        #print("{}".format(hex(buf)))
        buf = (buf * pow(16,4)) + self.reserve
        #print("{}".format(hex(buf)))
        buf = (buf * pow(16,4)) + self.datamount
        #print("{}".format(hex(buf)))
        buf = (buf * pow(16,4)) + self.datalength
        #print("{}".format(hex(buf)))
        buf = (buf * pow(16,14)) + self.datapos
        #print("{}".format(hex(buf)))
        return buf

class xgt_packet:
    header = xgt_header()
    body = xgt_body()
    
    def get_packet(self):
        buf = self.header.get_packet()
        buf = (buf * pow(16,len(hex(self.body.get_packet()))-2)) + self.body.get_packet()
        return buf
    


xgt_hd = xgt_header()
xgt_bd = xgt_body()
xgt_pk = xgt_packet()
#xgt_req = req_packet()
xgt_hd.get_bcc()
print("header : {} {} {}".format(hex(xgt_hd.get_packet()),type(xgt_hd.get_packet()),len(hex(xgt_hd.get_packet()))-2))
print("body : {} {}".format(hex(xgt_bd.get_packet()),type(xgt_bd.get_packet())))
print("packet : {} ".format(hex(xgt_pk.get_packet())))
#byte_obj = bytes.fromhex(xgt_hd.get_header())
#print("ascii : {}".format(byte_obj.decode("ASCII")))
#print("{}".format(xgt_req.getPacket()))
#print(("LSIS-XGT").encode('utf-8').hex())
#print(("%MW1000").encode('utf-8').hex())
#print(("%MW1000").encode('utf-8').hex())

#clientSocket = socket(AF_INET, SOCK_STREAM)  # 서버에 접속하기 위한 소켓을 생성한다.
def connect():  
    try:
        clientSocket.connect(ADDR)  # 서버에 접속을 시도한다.

    except  Exception as e:
        print('%s:%s' % ADDR)
        sys.exit()
    print('connect is success')

def datarecv():
    #sendData = b'4C5349532D58475400000000A0330200110003405400020000000100070025445731303030'
    print("send")
    #print(type(sendData))
    #print(sendData)
    #print(sendData.__sizeof__())
    #print(binascii.hexlify(sendData).decode("ascii"))
    clientSocket.sendall(hex(xgt_pk.get_packet()))
    recvData = clientSocket.recv(6000)
    print(type(recvData))
    print("recv")
    print(binascii.hexlify(recvData))
    print(recvData.__sizeof__())

#connect()
#datarecv()

#clientSocket.close()
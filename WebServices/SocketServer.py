'''
Created on 8 de fev de 2016

@author: root
'''
'''
Created on 7 de ago de 2015

@author: root
'''
import classificar_palavra
from socket import *
import os
import sys
import math
import socket #do pacote socket, importe tudo, inclusive a classe socket
import threading
import struct
import wave

class ClientThreads(threading.Thread):
    
    def __init__(self,HOST,PORT, socket, cont):
        threading.Thread.__init__(self)
        self.ip = HOST
        self.port = PORT
        self.socket = socket
        self.cont = cont
        print ("## New thread started for "+HOST+":"+str(PORT))
    
    def recvall(self, count):
        sock = self.socket
        buf = b''
        while count:
            newbuf = sock.recv(count)
            if not newbuf: return None
            buf += newbuf # + buf
            count -= len(newbuf)
        return buf
    
    def run(self):    
        print ("Connection from : "+self.ip+":"+str(self.port))
        
        #recebe o tamanho do audio em array de bytes e converte pra int
        size = self.recvall(4) #size
        size = struct.unpack("<i", size)[0]
        print("File size: ", size)
        
        #receber arquivo_audio e salvar em Sounds
        audioByte = self.recvall(size)
        data = audioByte
        
        if not audioByte:
            print('Audio not found')
       
        audioFilename = "Sounds/arquivo_audio%d.wav" % self.cont
        newFile = wave.open(audioFilename, mode='wb')
        newFile.setnchannels(1)
        newFile.setsampwidth(2)
        newFile.setframerate(8000)
        newFile.writeframesraw(data)
        newFile.close()
        
        #reconhece o audio
        palavra = classificar_palavra.classificar(audioFilename)
        print("Reconheceu " + palavra)
        #prepara o audio reconhecido para o envio        
        palavraBytes = bytes(palavra, 'UTF-8')
        lenBytes = len(palavraBytes).to_bytes(4, byteorder='little')
        #envia o tamanho do audio
        self.socket.sendall(lenBytes)
        
        #envia o audio
        self.socket.sendall(bytes(palavra, 'UTF-8'))
        
        os.remove(audioFilename)
        print ("Client disconnected...") 
        self.socket.close()

class SocketServer(threading.Thread):
    isAlive = True
    threads = []
    s = None
    
    def __init__(self):
        threading.Thread.__init__(self)
        try:
        #create an AF_INET, STREAM socket (TCP)
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except socket.error as msg:
            print ('Failed to creat socket.Error code: ' + str(msg[0]) + ' , Error message : ' + msg[1])
            sys.exit();
            
    def run(self):
        HOST = '0.0.0.0';#'0.0.0.0';
        PORT = 8080;    
        
        print ('Socket created')
        
        #Bind socket to local host and port
        try:
            self.s.bind((HOST, PORT))
        except socket.error as msg:
            print ('Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1])
            sys.exit()
        
        print ('Socket bind complete')
        
        #Start listening to the clients
        self.s.listen(10)
        print ('Now listening clients on ip={' + HOST + '} and port={' + str(PORT) + '}');
    
        c = 0
        while self.isAlive:
            #wait to accept a connection - blocking call
            (clientsocket, (HOST, PORT)) = self.s.accept()
            #start new thread   
            newthread = ClientThreads(HOST, PORT, clientsocket, c)
            newthread.start()
            self.threads.append(newthread)
            c+=1
        
        for t in self.threads:
            t.join()
            
        self.s.close()
        
    def Close(self):
        for t in self.threads:
            t.close()
        
        self.isAlive = False    
        self.s.close()
        
        
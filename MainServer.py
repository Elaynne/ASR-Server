'''
Created on 8 de fev de 2016

@author: root
'''
import WebServer
import SocketServer

if __name__ == "__main__":
    socketServer = SocketServer.SocketServer()
    socketServer.start()

import socket
import numpy as np
import json

HOST = "192.168.2.41"
PORT = 6850

class Socket_Util:
    def __init__(self):
        self.socket = None
        self.conn = None

    def init_socket(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((HOST, PORT))
        self.socket.listen(5)

        self.conn, address = self.socket.accept()
        print("Connection from: " + str(address))

    def send_data(self, data):
        self.conn.send(data)
        print("sent")

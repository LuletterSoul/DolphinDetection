import socket
import threading


class WebsocketServer(threading.Thread):
    def __init__(self, port):
        self.host = '127.0.0.1'
        self.port = port
        super(WebsocketServer, self).__init__()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(5)

    def run(self):
        while True:
            print('waiting client to connect...')
            conn, addr = self.sock.accept()
            print("client {} connect successfully:".format(addr))
            conn.send("welcome...".encode("utf8"))
            while True:
                try:
                    info = conn.recv(1024)
                    print(info.decode('utf-8'))
                except Exception as e:
                    print(e)


def main():
    websocket_server = WebsocketServer(3400)
    websocket_server.run()


if __name__ == '__main__':
    main()

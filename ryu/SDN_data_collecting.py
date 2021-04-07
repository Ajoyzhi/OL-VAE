# modify the switch monitor to get 
# (datapath, src_mac, dst_mac, packets_count, bytes_count) for flow entry
# (datapath, in_port, out_port, rx-pkts, rx-bytes, rx-error, tx-pkts, tx-bytes, tx-error) for port status


from operator import attrgetter

from ryu.app import simple_switch_13
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER, CONFIG_DISPATCHER,HANDSHAKE_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub

import csv
import time


class Ajoy_monitor(simple_switch_13.SimpleSwitch13):

    def __init__(self, *args, **kwargs):
        super(Ajoy_monitor, self).__init__(*args, **kwargs)
        self.datapaths = {}
        # after init, then run _monitor()
        self.monitor_thread = hub.spawn(self._monitor)

        # get data for flow entry
        self.fe_time = []
        self.fe_datapath = []
        self.in_port = []
        self.dst_mac = []
        self.packets_count = []
        self.bytes_count = []

        # get data for port
        self.port_time = []
        self.port_datapath = []
        self.port = []
        self.rx_packets = []
        self.rx_bytes = []
        self.tx_packets = []
        self.tx_bytes = []

        # get evn data
        self.echo_temp = {}
        self.fe_echo_delay = []
        self.port_echo_delay = []
        
        # csv path data
        self.port_path_pre = "/home/ajoy/Ajoy_data/"
        self.fe_path_pre = "/home/ajoy/Ajoy_data/"

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.debug('register datapath: %016x', datapath.id)
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.debug('unregister datapath: %016x', datapath.id)
                del self.datapaths[datapath.id]

    def _monitor(self):
        # collect data for 60s
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            # save flow entry data
            self.save_flow_entry_data()
            # save port status
            self.save_port_data()
            # clear the fe_list
            self.fe_time = []
            self.fe_datapath = []
            self.in_port = []
            self.dst_mac = []
            self.packets_count = []
            self.bytes_count = []
            self.fe_echo_delay = []

            # clear the port list
            self.port_time = []
            self.port_datapath = []
            self.port = []
            self.rx_packets = []
            self.rx_bytes = []
            self.tx_packets = []
            self.tx_bytes = []
            self.port_echo_delay = []
            hub.sleep(30)

    def _request_stats(self, datapath):
        self.logger.debug('send stats request: %016x', datapath.id)
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        # send flow state request message
        fe_req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(fe_req)
        
        # send port state request message 
        port_req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(port_req)
        
        # get env data: controller-switch delay
        real_time = "%.12f" % time.time()
        # encode():str->bytes
        echo_req = parser.OFPEchoRequest(datapath, data=real_time.encode())
        datapath.send_msg(echo_req)
        hub.sleep(0.05)

    # get flow entry statistic
    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        body = ev.msg.body
        self.logger.info('datapath         '
                         'in-port eth-dst           '
                         'out-port packets  bytes')
        self.logger.info('---------------- '
                         '-------- ----------------- '
                         '-------- -------- --------')
        # add the function to get table-miss flow entry
        for stat in body:
            datapath = ev.msg.datapath.id
            # table-miss flow entry
            if stat.priority == 0:
                in_port = 0
                eth_dst = 0
                out_port = 0
                packets_count = stat.packet_count
                bytes_count = stat.byte_count
            # normal flow entry
            else:
                in_port = stat.match['in_port']
                eth_dst = stat.match['eth_dst']
                out_port = stat.instructions[0].actions[0].port
                packets_count = stat.packet_count
                bytes_count = stat.byte_count

            self.logger.info('%016x %8x %17s %8x %8d %8d',
                        datapath, in_port, eth_dst, out_port, packets_count, bytes_count)

            # get data datapath, src_mac(in_port), dst_mac, packets_count, bytes_count
            real_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            self.fe_time.append(real_time)
            self.fe_datapath.append(datapath)
            self.in_port.append(in_port)
            self.dst_mac.append(eth_dst)
            self.packets_count.append(packets_count)
            self.bytes_count.append(bytes_count)

    # get port stats
    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        body = ev.msg.body
        self.logger.info('datapath         port     '
                         'rx-pkts  rx-bytes rx-error '
                         'tx-pkts  tx-bytes tx-error')
        self.logger.info('---------------- -------- '
                         '-------- -------- -------- '
                         '-------- -------- --------')
        for stat in sorted(body, key=attrgetter('port_no')):
            self.logger.info('%016x %8x %8d %8d %8d %8d %8d %8d',
                             ev.msg.datapath.id, stat.port_no,
                             stat.rx_packets, stat.rx_bytes, stat.rx_errors,
                             stat.tx_packets, stat.tx_bytes, stat.tx_errors)
            # get (real_time, datapath, port, rx_packets, rx_byte, tx_packets, tx_byte)
            real_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            self.port_time.append(real_time)
            self.port_datapath.append(ev.msg.datapath.id)
            self.port.append(stat.port_no)
            self.rx_packets.append(stat.rx_packets)
            self.rx_bytes.append(stat.rx_bytes)
            self.tx_packets.append(stat.tx_packets)
            self.tx_bytes.append(stat.tx_bytes)

    # get controller-switch delay
    @set_ev_cls(ofp_event.EventOFPEchoReply, [MAIN_DISPATCHER,CONFIG_DISPATCHER,HANDSHAKE_DISPATCHER])
    def echo_reply_handler(self, ev):
        data = ev.msg.data
        dp_id = ev.msg.datapath.id
        try:
            delay = time.time() - eval(data)
            self.echo_temp[dp_id] = delay
            self.logger.info("datapath_id:%d" % dp_id + "delay:%.5f" % delay)
        except Exception as e:
            self.logger.info("when switch %d" % dp_id + "reply echo, ERROR!!")
            
    def save_flow_entry_data(self):
        for dp in self.fe_datapath:
            # self.logger.info("datapath in fe_datapath:%d" % dp)
            self.fe_echo_delay.append(self.echo_temp[dp])
        # flow entry data
        flow_entry_data = zip(self.fe_time, self.fe_datapath, self.in_port, self.dst_mac, self.packets_count, self.bytes_count, self.fe_echo_delay)
        # flow_entry_header = ['time', 'datapath', 'in_port', 'dst_mac', 'packets_count', 'bytes_count', 'delay']
        for dp in self.datapaths:
            fe_path = self.fe_path_pre + "s" + str(dp) + "_fe.csv"
            fe_file = open(fe_path, 'a')
            fe_writer = csv.writer(fe_file, dialect='excel')
            for item in flow_entry_data:
                if item[1] == dp:
                    fe_writer.writerow(item)
            fe_file.close()
        
    def save_port_data(self):
        for dp in self.port_datapath:
            # self.logger.info("datapath in port_datapath:%d" % dp)
            self.port_echo_delay.append(self.echo_temp[dp])
        # port status
        port_data = zip(self.port_time, self.port_datapath, self.port, self.rx_packets, self.rx_bytes, self.tx_packets, self.tx_bytes, self.port_echo_delay)
        # port_header = ['time', 'datapath', 'port', 'rx_packets', 'rx_bytes', 'tx_packets', 'tx_bytes', 'delay']
        for dp in self.datapaths:
            port_path = self.port_path_pre + "s" + str(dp) + "_port.csv"
            port_file = open(port_path, 'a')
            port_writer = csv.writer(port_file, dialect='excel')
            for item in port_data:
                if item[1] == dp:
                    port_writer.writerow(item)
            port_file.close()

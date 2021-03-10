# modify the switch monitor to get 
# (datapath, src_mac, dst_mac, packets_count, bytes_count) for flow entry
# (datapath, in_port, out_port, rx-pkts, rx-bytes, rx-error, tx-pkts, tx-bytes, tx-error) for port status
# 2021.3.9 put all the data into cvs file at ~/Ajoy_data

from operator import attrgetter

from ryu.app import simple_switch_13
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub

import csv
import os


class SimpleMonitor13(simple_switch_13.SimpleSwitch13):

    def __init__(self, *args, **kwargs):
        super(SimpleMonitor13, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor)
        # get data for flow entry
        self.fe_datapath = []
        self.in_port = []
        self.dst_mac = []
        self.packets_count = []
        self.bytes_count = []
        # get data for port 
        self.port_datapath = []
        self.port = []
        self.rx_packets = []
        self.rx_bytes = []
        self.tx_packets = []
        self.tx_bytes = []
        # get evn data
        self.switch_buffer = []
        self.lantency = []
        self.controller_buffet = []
        
        # csv path data
        self.port_path = "/home/ajoy/Ajoy_data/port_data.csv"
        self.flow_entry_path = "/home/ajoy/Ajoy_data/fe_data.csv"

    @set_ev_cls(ofp_event.EventOFPStateChange,
                [MAIN_DISPATCHER, DEAD_DISPATCHER])
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
        # get 10 * 10s 
        for i in range(5):
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(10)
        # after  getting all the data, putting them into file
        # save flow entry data
        self.save_flow_entry_data()
        # save port status
        self.save_port_data()

        """
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(10)
        """
    def _request_stats(self, datapath):
        self.logger.debug('send stats request: %016x', datapath.id)
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        # send flow state request message
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)
        
        # send port state request message 
        req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(req)
        
        # get env data: switch buffer, traffic lantency(epoch), controller buffer

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        body = ev.msg.body
        
        self.logger.info('datapath         '
                         'in-port eth-dst           '
                         'out-port packets  bytes')
        self.logger.info('---------------- '
                         '-------- ----------------- '
                         '-------- -------- --------')
                     
        # no table-miss flow entry
        for stat in sorted([flow for flow in body if flow.priority == 1],
                           key=lambda flow: (flow.match['in_port'],
                                             flow.match['eth_dst'])):
            self.logger.info('%016x %8x %17s %8x %8d %8d',
                             ev.msg.datapath.id,
                             stat.match['in_port'], stat.match['eth_dst'],
                             stat.instructions[0].actions[0].port,
                             stat.packet_count, stat.byte_count)
                             
            # get datapath, src_mac(in_port), dst_mac, packets_count, bytes_count
            self.fe_datapath.append(ev.msg.datapath.id)
            self.in_port.append(stat.match['in_port'])
            self.dst_mac.append(stat.match['eth_dst'])
            self.packets_count.append(stat.packet_count)
            self.bytes_count.append(stat.byte_count)

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
            # get (datapath, port, rx_packets, rx_byte, tx_packets, tx_byte)
            self.port_datapath.append(ev.msg.datapath.id)
            self.port.append(stat.port_no)
            self.rx_packets.append(stat.rx_packets)
            self.rx_bytes.append(stat.rx_bytes)
            self.tx_packets.append(stat.tx_packets)
            self.tx_bytes.append(stat.tx_bytes)
            
    def save_flow_entry_data(self):
        # flow entry data
        flow_entry_data = zip(self.fe_datapath, self.in_port, self.dst_mac, self.packets_count, self.bytes_count)
        flow_entry_header = ['datapath', 'in_port', 'dst_mac', 'packets_count', 'bytes_count']
        """
        # judge if existing the file;if not, create
        if not os.path.exists(self.flow_entry_path):
            os.mknod(self.flow_entry_path)
        """
        flow_entry_file = open(self.flow_entry_path, 'a', newline='')
        fe_writer = csv.writer(flow_entry_file, dialect='excel')
        fe_writer.writerow(flow_entry_header)
        for item in flow_entry_data:
            fe_writer.writerow(item)
        flow_entry_file.close()
        
    def save_port_data(self):
        # port status
        port_data = zip(self.port_datapath, self.port, self.rx_packets, self.rx_bytes, self.tx_packets, self.tx_bytes)
        port_header = ['datapath', 'port', 'rx_packets', 'rx_bytes', 'tx_packets', 'tx_bytes']
        """
        if not os.path.exists(self.port_path):
            os.mknod(self.port_path)
        """
        port_file = open(self.port_path, 'a', newline='')
        port_writer = csv.writer(port_file, dialect='excel')
        port_writer.writerow(port_header)
        for item in port_data:
            port_writer.writerow(item)
        port_file.close()     

# -*- coding: utf-8 -*-
# Ryu Controller - 4-slice SDN  (Python 2.7 / Ryu 4.31 compatible)
# Slices: URLLC (1:10) | URLLC_eMBB_MIX (1:11) | eMBB (1:12) | mMTC (1:13)
#
# Endpoints consumed by PPO agent:
#   GET  /getports              -> list of registered interfaces
#   GET  /getstats              -> latest per-slice stats forwarded from Mininet
#   GET  /getenergy             -> proxy energy metric (active-slice ratio)
#   POST /register_interfaces
#   POST /monitoring            -> receives tc stats from Mininet topology
#   POST /setaction             -> receives rate allocation from PPO agent

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import (CONFIG_DISPATCHER, MAIN_DISPATCHER,
                                     DEAD_DISPATCHER, set_ev_cls)
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types
from ryu.lib import hub
from ryu.app.wsgi import WSGIApplication, ControllerBase, route
from webob import Response
import json
import time

# ---------------------------------------------------------------------------
# Slice meta-data  (must match topology.py and NetworkConfig)
# ---------------------------------------------------------------------------
SLICE_NAMES = ['URLLC', 'URLLC_eMBB_MIX', 'eMBB', 'mMTC']
TC_CLASSES  = ['1:10', '1:11', '1:12', '1:13']
NUM_SLICES  = 4

SLA_LATENCY = {
    'URLLC':          1.0,
    'URLLC_eMBB_MIX': 2.0,
    'eMBB':           10.0,
    'mMTC':           50.0,
}

MAX_BW_MBPS = 750

# Simple power model for simulation. Replace with measured hardware values for real experiments.
BASE_POWER_W = 20.0
SLICE_POWER_W = {
    'URLLC': 10.0,
    'URLLC_eMBB_MIX': 8.0,
    'eMBB': 15.0,
    'mMTC': 5.0,
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _cls_to_slice(cls):
    mapping = {
        '1:10': 'URLLC',
        '1:11': 'URLLC_eMBB_MIX',
        '1:12': 'eMBB',
        '1:13': 'mMTC',
    }
    return mapping.get(cls, 'unknown')


def _log_monitoring_summary(dpid, iface_stats, timestamp):
    print("\n[MONITORING] DPID=%s  t=%s" % (dpid, time.ctime(timestamp)))
    print("  %-14s %-6s %-16s %8s %8s %8s %8s" % (
        'Interface', 'Class', 'Slice', 'Drop', 'Rate', 'Tput', 'Lat'))
    print("  " + "-" * 72)
    for s in iface_stats:
        cls        = s.get('class', '?')
        slice_name = s.get('slice', _cls_to_slice(cls))
        print("  %-14s %-6s %-16s %8.4f %6dMb %8.3f %8.3fms" % (
            s.get('interface', '?'),
            cls,
            slice_name,
            s.get('dropped', 0),
            s.get('rate', 0),
            s.get('throughput', 0),
            s.get('latency', 0),
        ))


# ---------------------------------------------------------------------------
# REST API controller
# ---------------------------------------------------------------------------

class InterfaceAPI(ControllerBase):
    switch_interfaces = {}
    monitoring_data   = {}
    previous_stats    = {}
    stats_todrl       = []
    new_rates         = []
    active_counts     = dict((name, 0) for name in SLICE_NAMES)
    total_intervals   = 0
    current_power_w   = BASE_POWER_W
    total_energy_j    = 0.0
    last_energy_timestamp = None
    ports             = []

    def __init__(self, req, link, data, **config):
        super(InterfaceAPI, self).__init__(req, link, data, **config)

    @staticmethod
    def _json_ok(body_dict):
        return Response(
            content_type='application/json',
            body=json.dumps(body_dict),
        )

    @staticmethod
    def _json_err(body_dict, status=400):
        return Response(
            status=status,
            content_type='application/json',
            body=json.dumps(body_dict),
        )

    @staticmethod
    def _server_error(exc):
        return Response(status=500, body=str(exc))

    @route('interface', '/register_interfaces', methods=['POST'])
    def register_interfaces(self, req, **kwargs):
        try:
            data       = req.json if req.body else {}
            dpid       = int(data.get('dpid'))
            interfaces = data.get('interfaces', [])
            InterfaceAPI.ports.append(interfaces)
            InterfaceAPI.switch_interfaces[dpid] = interfaces
            print("[API] Interfaces enregistrees pour DPID %s: %s" % (dpid, interfaces))
            return self._json_ok({'status': 'ok'})
        except Exception as e:
            print("[API ERROR] register_interfaces: %s" % str(e))
            return self._server_error(e)

    @route('interface', '/getports', methods=['GET'])
    def get_ports(self, req, **kwargs):
        if not InterfaceAPI.ports:
            time.sleep(3)
        return self._json_ok({'ports': InterfaceAPI.ports})

    @route('interface', '/monitoring', methods=['POST'])
    def monitoring(self, req, **kwargs):
        try:
            data        = req.json if req.body else {}
            dpid        = int(data.get('dpid'))
            timestamp   = data.get('timestamp', time.time())
            iface_stats = data.get('stats', [])

            InterfaceAPI.monitoring_data[dpid] = {
                'timestamp':  timestamp,
                'interfaces': iface_stats,
            }
            InterfaceAPI.total_intervals += 1

            if dpid not in InterfaceAPI.previous_stats:
                InterfaceAPI.previous_stats[dpid] = {}

            enriched = []
            active_slices = set()
            for stat in iface_stats:
                cls        = stat.get('class', '')
                iface      = stat.get('interface', '')
                slice_name = stat.get('slice', _cls_to_slice(cls))
                rate       = stat.get('rate', 0)
                latency    = stat.get('latency', 0)
                throughput = stat.get('throughput', 0)
                drop_rate  = stat.get('dropped', 0)
                parent_rate        = stat.get('parent_rate', 0) * 1000000
                nbre_demands       = stat.get('nbre_demands', 0)
                nbre_demands_bytes = stat.get('nbre_demands_bytes', 0)
                sla_latency        = stat.get('sla_latency',
                                              SLA_LATENCY.get(slice_name, 10.0))

                if rate > 0:
                    active_slices.add(slice_name)
                    InterfaceAPI.active_counts[slice_name] = (
                        InterfaceAPI.active_counts.get(slice_name, 0) + 1
                    )

                enriched.append({
                    'timestamp':          timestamp,
                    'dpid':               dpid,
                    'interface':          iface,
                    'class':              cls,
                    'slice':              slice_name,
                    'dropped':            drop_rate,
                    'latency':            latency,
                    'throughput':         throughput,
                    'rate':               rate,
                    'parent_rate':        parent_rate,
                    'nbre_demands':       nbre_demands,
                    'nbre_demands_bytes': nbre_demands_bytes,
                    'sla_latency':        sla_latency,
                })

            power_w = BASE_POWER_W
            for slice_name in active_slices:
                power_w += SLICE_POWER_W.get(slice_name, 0.0)

            now = float(timestamp)
            if InterfaceAPI.last_energy_timestamp is not None:
                elapsed = max(0.0, now - InterfaceAPI.last_energy_timestamp)
                InterfaceAPI.total_energy_j += power_w * elapsed

            InterfaceAPI.current_power_w = power_w
            InterfaceAPI.last_energy_timestamp = now

            InterfaceAPI.stats_todrl = enriched
            _log_monitoring_summary(dpid, iface_stats, timestamp)

            if InterfaceAPI.new_rates:
                pending = list(InterfaceAPI.new_rates)
                InterfaceAPI.new_rates = []
                print("[API] Retour des nouveaux rates vers Mininet: %s" % pending)
                return self._json_ok({
                    'status':    'ok',
                    'message':   'Monitoring data received - rates updated',
                    'new_rates': pending,
                })

            return self._json_ok({
                'status':  'ok',
                'message': 'Monitoring data received',
            })

        except Exception as e:
            print("[API ERROR] monitoring: %s" % str(e))
            return self._server_error(e)

    @route('interface', '/getstats', methods=['GET'])
    def getstats(self, req, **kwargs):
        try:
            snap = InterfaceAPI.stats_todrl
            if snap:
                return self._json_ok({'status': 'ok', 'stats': snap})
            return self._json_ok({'status': 'error', 'message': 'Stats indisponibles'})
        except Exception as e:
            print("[API ERROR] getstats: %s" % str(e))
            return self._server_error(e)

    @route('interface', '/getenergy', methods=['GET'])
    def getenergy(self, req, **kwargs):
        try:
            return self._json_ok({
                'status': 'ok',
                'power_w': InterfaceAPI.current_power_w,
                'energy_joules': InterfaceAPI.total_energy_j,
            })
        except Exception as e:
            print("[API ERROR] getenergy: %s" % str(e))
            return self._server_error(e)

    @route('interface', '/setaction', methods=['POST'])
    def setaction(self, req, **kwargs):
        try:
            data  = req.json if req.body else {}
            rates = data.get('rates', [])
            if len(rates) != NUM_SLICES:
                return self._json_err({
                    'status':  'error',
                    'message': 'Expected %d rates, got %d' % (NUM_SLICES, len(rates)),
                })
            data['rates'] = [max(0.0, min(float(r), float(MAX_BW_MBPS)))
                             for r in rates]
            InterfaceAPI.new_rates.append(data)
            print("[API] Nouvelle allocation recue: %s" % data)
            return self._json_ok({'status': 'ok'})
        except Exception as e:
            print("[API ERROR] setaction: %s" % str(e))
            return self._server_error(e)


# ---------------------------------------------------------------------------
# Ryu application
# ---------------------------------------------------------------------------

class SwitchWithStats(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {'wsgi': WSGIApplication}

    def __init__(self, *args, **kwargs):
        super(SwitchWithStats, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.datapaths   = {}

        wsgi = kwargs['wsgi']
        wsgi.register(InterfaceAPI)

        self.switch_interfaces = InterfaceAPI.switch_interfaces
        self.monitoring_data   = InterfaceAPI.monitoring_data

        self.logger.info("Controleur SDN 4-slices initialise")
        self.monitor_thread = hub.spawn(self._periodic_analysis)

    def _periodic_analysis(self):
        while True:
            hub.sleep(2)
            if not self.monitoring_data:
                continue
            for dpid, data in self.monitoring_data.items():
                self._check_sla_violations(dpid, data['interfaces'])

    def _check_sla_violations(self, dpid, iface_stats):
        for stat in iface_stats:
            cls        = stat.get('class', '')
            slice_name = stat.get('slice', _cls_to_slice(cls))
            latency    = stat.get('latency', 0)
            sla        = stat.get('sla_latency', SLA_LATENCY.get(slice_name, 10.0))
            drop_rate  = stat.get('dropped', 0)
            if latency > sla:
                self.logger.warning(
                    "[SLA VIOLATION] DPID=%s %s %s(%s): lat=%.2fms > SLA=%.2fms drop=%.4f",
                    dpid,
                    stat.get('interface', '?'),
                    cls,
                    slice_name,
                    latency,
                    sla,
                    drop_rate,
                )

    @set_ev_cls(ofp_event.EventOFPStateChange,
                [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.info('Datapath enregistre: %016x', datapath.id)
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.info('Datapath supprime: %016x', datapath.id)
                del self.datapaths[datapath.id]

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto  = datapath.ofproto
        parser   = datapath.ofproto_parser

        match   = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self._add_flow(datapath, priority=0, match=match, actions=actions)
        self._install_slice_flows(datapath)

    def _install_slice_flows(self, datapath):
        """
        Install OpenFlow rules matching DSCP for 4-slice priority forwarding.
          URLLC          DSCP 0x0A  (ToS 0x28)  prio 40
          URLLC_eMBB_MIX DSCP 0x06  (ToS 0x18)  prio 30
          eMBB           DSCP 0x04  (ToS 0x10)  prio 20
          mMTC           DSCP 0x14  (ToS 0x50)  prio 10
        """
        ofproto = datapath.ofproto
        parser  = datapath.ofproto_parser

        slice_flows = [
            ('URLLC',          0x0A, 40),
            ('URLLC_eMBB_MIX', 0x06, 30),
            ('eMBB',           0x04, 20),
            ('mMTC',           0x14, 10),
        ]
        for slice_name, dscp, prio in slice_flows:
            match   = parser.OFPMatch(ip_dscp=dscp, eth_type=0x0800)
            actions = [parser.OFPActionOutput(ofproto.OFPP_NORMAL)]
            self._add_flow(datapath, priority=prio, match=match, actions=actions)
            self.logger.info("Flow installe: %s DSCP=0x%02x prio=%d",
                             slice_name, dscp, prio)

    def _add_flow(self, datapath, priority, match, actions, buffer_id=None):
        ofproto = datapath.ofproto
        parser  = datapath.ofproto_parser
        inst    = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                                actions)]
        kw = dict(datapath=datapath, priority=priority,
                  match=match, instructions=inst)
        if buffer_id is not None:
            kw['buffer_id'] = buffer_id
        datapath.send_msg(parser.OFPFlowMod(**kw))

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg      = ev.msg
        datapath = msg.datapath
        dpid     = datapath.id
        ofproto  = datapath.ofproto
        parser   = datapath.ofproto_parser
        in_port  = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        if eth.ethertype in (ether_types.ETH_TYPE_LLDP,
                             ether_types.ETH_TYPE_IPV6):
            return

        dst = eth.dst
        src = eth.src

        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port
        out_port = self.mac_to_port[dpid].get(dst, ofproto.OFPP_FLOOD)

        actions = [parser.OFPActionOutput(out_port)]

        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                self._add_flow(datapath, priority=1, match=match,
                               actions=actions, buffer_id=msg.buffer_id)
                return
            else:
                self._add_flow(datapath, priority=1, match=match,
                               actions=actions)

        data = msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None
        out  = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=data,
        )
        datapath.send_msg(out)

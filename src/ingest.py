#!/usr/bin/env python3
"""
Aegis Sentinel - Passive Unidirectional Network Ingest Engine
===================================================================

This module provides a strictly passive, read-only network flow ingestion
and metadata extraction pipeline using Scapy. It operates in offline mode
over PCAP/PCAPNG capture files (or passive network taps/diodes) with zero
packet transmission, zero active probing, and zero return packets.

Features Extracted:
- Unidirectional 5-tuple flow identification (Src IP, Dst IP, Src Port, Dst Port, Protocol)
- Flow Duration & Temporal Metrics (Start time, End time, Duration in seconds)
- Flow Throughput & Rates (Bytes/sec, Packets/sec, Payload Bytes/sec)
- Inter-Arrival Time (IAT) Statistics (Mean, Std Dev, Min, Max, Total)
- Packet Length Distributions (Mean, Std Dev, Min, Max, Total Wire & Payload Bytes)
- TCP State & Control Flags (SYN, ACK, FIN, RST, PSH, URG, ECE, CWR, Window sizes)
- Application/ICMP Attributes & Anomaly Indicators (Single packet flows, Header/Payload ratios)

Architecture Guarantee:
- Purely read-only / passive: Scapy PcapReader streams bytes from local disk.
- Zero network socket emission: No ARP, DNS, ICMP, TCP, or UDP frames are transmitted.
- Strict unidirectional tracking: Forward (A->B) and reverse (B->A) flows are treated
  as isolated unidirectional streams by default, mirroring data diode taps.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import scapy layers in a safe, passive manner
from scapy.layers.inet import ICMP, IP, TCP, UDP
from scapy.layers.inet6 import IPv6
from scapy.packet import Packet
from scapy.utils import PcapReader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("aegis.ingest")

# Protocol number mapping
PROTO_MAP = {
    1: "ICMP",
    6: "TCP",
    17: "UDP",
    41: "IPv6-Route",
    47: "GRE",
    50: "ESP",
    51: "AH",
    58: "IPv6-ICMP",
    89: "OSPF",
    132: "SCTP",
}


@dataclass(frozen=True)
class FlowKey:
    """Represents the 5-tuple key for a unidirectional network flow."""
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str

    def to_string(self) -> str:
        return f"{self.src_ip}:{self.src_port}->{self.dst_ip}:{self.dst_port}_{self.protocol}"

    def to_bidirectional(self) -> FlowKey:
        """Returns a canonical bidirectional key (sorted endpoint pairs)."""
        ep1 = (self.src_ip, self.src_port)
        ep2 = (self.dst_ip, self.dst_port)
        if ep1 <= ep2:
            return FlowKey(self.src_ip, self.dst_ip, self.src_port, self.dst_port, self.protocol)
        else:
            return FlowKey(self.dst_ip, self.src_ip, self.dst_port, self.src_port, self.protocol)


@dataclass
class FlowAccumulator:
    """Accumulates packet attributes and computes flow metadata statistics."""
    key: FlowKey
    start_time: float
    end_time: float
    packet_count: int = 0
    total_wire_bytes: int = 0
    total_payload_bytes: int = 0
    
    # Inter-arrival times and packet lengths
    timestamps: List[float] = field(default_factory=list)
    packet_lengths: List[int] = field(default_factory=list)
    payload_lengths: List[int] = field(default_factory=list)
    
    # TCP specific metrics
    syn_count: int = 0
    ack_count: int = 0
    fin_count: int = 0
    rst_count: int = 0
    psh_count: int = 0
    urg_count: int = 0
    cwr_count: int = 0
    ece_count: int = 0
    tcp_windows: List[int] = field(default_factory=list)
    initial_window: int = 0
    
    # ICMP specific metrics
    icmp_types: List[int] = field(default_factory=list)
    icmp_codes: List[int] = field(default_factory=list)

    def add_packet(self, pkt_time: float, wire_len: int, payload_len: int, pkt: Packet) -> None:
        """Updates flow state with an incoming passively observed packet."""
        if self.packet_count == 0:
            self.start_time = pkt_time
            self.end_time = pkt_time
        else:
            self.end_time = max(self.end_time, pkt_time)
            
        self.packet_count += 1
        self.total_wire_bytes += wire_len
        self.total_payload_bytes += payload_len
        
        self.timestamps.append(pkt_time)
        self.packet_lengths.append(wire_len)
        self.payload_lengths.append(payload_len)

        # Inspect TCP Flags & Window
        if pkt.haslayer(TCP):
            tcp_layer = pkt[TCP]
            flags = int(tcp_layer.flags)
            if flags & 0x02:  # SYN
                self.syn_count += 1
            if flags & 0x10:  # ACK
                self.ack_count += 1
            if flags & 0x01:  # FIN
                self.fin_count += 1
            if flags & 0x04:  # RST
                self.rst_count += 1
            if flags & 0x08:  # PSH
                self.psh_count += 1
            if flags & 0x20:  # URG
                self.urg_count += 1
            if flags & 0x40:  # ECE
                self.ece_count += 1
            if flags & 0x80:  # CWR
                self.cwr_count += 1
                
            win = int(getattr(tcp_layer, "window", 0))
            self.tcp_windows.append(win)
            if self.packet_count == 1:
                self.initial_window = win

        # Inspect ICMP
        elif pkt.haslayer(ICMP):
            icmp_layer = pkt[ICMP]
            self.icmp_types.append(int(getattr(icmp_layer, "type", 0)))
            self.icmp_codes.append(int(getattr(icmp_layer, "code", 0)))

    def to_record(self) -> Dict[str, Any]:
        """Calculates final mathematical features and statistical metrics."""
        duration = max(0.0, self.end_time - self.start_time)
        
        # Inter-arrival times (IAT)
        if len(self.timestamps) > 1:
            # Sort timestamps just in case out-of-order packets occurred in capture
            sorted_times = sorted(self.timestamps)
            iats = np.diff(sorted_times)
            iat_mean = float(np.mean(iats))
            iat_std = float(np.std(iats, ddof=1)) if len(iats) > 1 else 0.0
            iat_min = float(np.min(iats))
            iat_max = float(np.max(iats))
            iat_total = float(np.sum(iats))
        else:
            iat_mean = 0.0
            iat_std = 0.0
            iat_min = 0.0
            iat_max = 0.0
            iat_total = 0.0

        # Packet Length Statistics
        if self.packet_lengths:
            pkt_len_mean = float(np.mean(self.packet_lengths))
            pkt_len_std = float(np.std(self.packet_lengths, ddof=1)) if len(self.packet_lengths) > 1 else 0.0
            pkt_len_min = int(np.min(self.packet_lengths))
            pkt_len_max = int(np.max(self.packet_lengths))
        else:
            pkt_len_mean = 0.0
            pkt_len_std = 0.0
            pkt_len_min = 0
            pkt_len_max = 0

        # Payload Length Statistics
        if self.payload_lengths:
            payload_len_mean = float(np.mean(self.payload_lengths))
            payload_len_std = float(np.std(self.payload_lengths, ddof=1)) if len(self.payload_lengths) > 1 else 0.0
            payload_len_max = int(np.max(self.payload_lengths))
            payload_len_min = int(np.min(self.payload_lengths))
        else:
            payload_len_mean = 0.0
            payload_len_std = 0.0
            payload_len_max = 0
            payload_len_min = 0

        # Rates (per second)
        # Avoid division by zero: if duration == 0 for single packet or microsecond burst,
        # rate is calculated over effective duration or 0.0
        effective_duration = duration if duration > 1e-6 else 0.0
        bytes_per_sec = (self.total_wire_bytes / duration) if duration > 1e-6 else 0.0
        pkts_per_sec = (self.packet_count / duration) if duration > 1e-6 else 0.0
        payload_bytes_per_sec = (self.total_payload_bytes / duration) if duration > 1e-6 else 0.0

        # TCP Window Stats
        if self.tcp_windows:
            tcp_win_mean = float(np.mean(self.tcp_windows))
            tcp_win_min = int(np.min(self.tcp_windows))
            tcp_win_max = int(np.max(self.tcp_windows))
        else:
            tcp_win_mean = 0.0
            tcp_win_min = 0
            tcp_win_max = 0

        # Ratios and threat indicators
        header_overhead = max(0, self.total_wire_bytes - self.total_payload_bytes)
        payload_ratio = (self.total_payload_bytes / self.total_wire_bytes) if self.total_wire_bytes > 0 else 0.0

        return {
            "flow_id": self.key.to_string(),
            "src_ip": self.key.src_ip,
            "dst_ip": self.key.dst_ip,
            "src_port": self.key.src_port,
            "dst_port": self.key.dst_port,
            "protocol": self.key.protocol,
            "start_time": f"{self.start_time:.6f}",
            "end_time": f"{self.end_time:.6f}",
            "flow_duration_sec": round(duration, 6),
            "packet_count": self.packet_count,
            "total_bytes": self.total_wire_bytes,
            "total_payload_bytes": self.total_payload_bytes,
            "bytes_per_sec": round(bytes_per_sec, 4),
            "pkts_per_sec": round(pkts_per_sec, 4),
            "payload_bytes_per_sec": round(payload_bytes_per_sec, 4),
            "iat_mean": round(iat_mean, 6),
            "iat_std": round(iat_std, 6),
            "iat_min": round(iat_min, 6),
            "iat_max": round(iat_max, 6),
            "iat_total": round(iat_total, 6),
            "pkt_len_mean": round(pkt_len_mean, 4),
            "pkt_len_std": round(pkt_len_std, 4),
            "pkt_len_min": pkt_len_min,
            "pkt_len_max": pkt_len_max,
            "payload_len_mean": round(payload_len_mean, 4),
            "payload_len_std": round(payload_len_std, 4),
            "payload_len_min": payload_len_min,
            "payload_len_max": payload_len_max,
            "payload_ratio": round(payload_ratio, 4),
            "syn_count": self.syn_count,
            "ack_count": self.ack_count,
            "fin_count": self.fin_count,
            "rst_count": self.rst_count,
            "psh_count": self.psh_count,
            "urg_count": self.urg_count,
            "cwr_count": self.cwr_count,
            "ece_count": self.ece_count,
            "tcp_win_init": self.initial_window,
            "tcp_win_mean": round(tcp_win_mean, 2),
            "tcp_win_min": tcp_win_min,
            "tcp_win_max": tcp_win_max,
            "icmp_type": self.icmp_types[0] if self.icmp_types else -1,
            "icmp_code": self.icmp_codes[0] if self.icmp_codes else -1,
            "is_single_packet": 1 if self.packet_count == 1 else 0,
        }


class PassiveFlowExtractor:
    """
    Streaming extractor that reads packets from a PCAP file passively
    and aggregates unidirectional flow metrics.
    """

    def __init__(
        self,
        pcap_path: str,
        unidirectional: bool = True,
        idle_timeout: Optional[float] = None,
    ) -> None:
        """
        Args:
            pcap_path: Path to the .pcap or .pcapng file.
            unidirectional: If True (default), flows are strictly segregated by direction (A->B != B->A).
            idle_timeout: Inactivity duration in seconds after which an existing flow is retired.
        """
        if not os.path.exists(pcap_path):
            raise FileNotFoundError(f"PCAP file not found: {pcap_path}")
            
        self.pcap_path = pcap_path
        self.unidirectional = unidirectional
        self.idle_timeout = idle_timeout
        self.active_flows: Dict[FlowKey, FlowAccumulator] = {}
        self.total_packets_processed = 0
        self.total_ip_packets = 0
        self.total_non_ip_packets = 0

    def _extract_packet_metadata(self, pkt: Packet) -> Optional[Tuple[FlowKey, float, int, int]]:
        """Extracts 5-tuple, timestamp, wire length, and payload length from a Scapy packet."""
        pkt_time = float(getattr(pkt, "time", time.time()))
        wire_len = len(pkt)
        payload_len = 0

        # Check IP layer (IPv4 or IPv6)
        src_ip = None
        dst_ip = None
        proto_name = "OTHER"

        if pkt.haslayer(IP):
            ip_layer = pkt[IP]
            src_ip = str(ip_layer.src)
            dst_ip = str(ip_layer.dst)
            proto_num = int(ip_layer.proto)
            proto_name = PROTO_MAP.get(proto_num, f"IP_{proto_num}")
        elif pkt.haslayer(IPv6):
            ip_layer = pkt[IPv6]
            src_ip = str(ip_layer.src)
            dst_ip = str(ip_layer.dst)
            proto_num = int(getattr(ip_layer, "nh", 0))
            proto_name = PROTO_MAP.get(proto_num, f"IPv6_{proto_num}")
        else:
            # Non-IP layer (ARP, LLDP, STP, Raw Ethernet, etc.)
            self.total_non_ip_packets += 1
            return None

        self.total_ip_packets += 1

        # Port extraction
        src_port = 0
        dst_port = 0

        if pkt.haslayer(TCP):
            tcp_layer = pkt[TCP]
            src_port = int(tcp_layer.sport)
            dst_port = int(tcp_layer.dport)
            proto_name = "TCP"
            payload_len = len(tcp_layer.payload)
        elif pkt.haslayer(UDP):
            udp_layer = pkt[UDP]
            src_port = int(udp_layer.sport)
            dst_port = int(udp_layer.dport)
            proto_name = "UDP"
            payload_len = len(udp_layer.payload)
        elif pkt.haslayer(ICMP):
            proto_name = "ICMP"
            payload_len = len(pkt[ICMP].payload)
        else:
            if hasattr(ip_layer, "payload"):
                payload_len = len(ip_layer.payload)

        key = FlowKey(
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=src_port,
            dst_port=dst_port,
            protocol=proto_name,
        )

        if not self.unidirectional:
            key = key.to_bidirectional()

        return key, pkt_time, wire_len, payload_len

    def extract_flows(self) -> List[Dict[str, Any]]:
        """
        Passively streams the PCAP file and aggregates all flows.
        Returns a list of extracted flow records as dictionaries.
        """
        logger.info(f"Opening PCAP in passive read-only streaming mode: {self.pcap_path}")
        logger.info(f"Architecture Mode: {'Unidirectional (Data Diode / Passive Tap)' if self.unidirectional else 'Bidirectional'}")
        
        expired_records: List[Dict[str, Any]] = []
        
        # Scapy PcapReader streams packets iteratively without loading entire file into memory
        with PcapReader(self.pcap_path) as reader:
            for pkt in reader:
                self.total_packets_processed += 1
                
                meta = self._extract_packet_metadata(pkt)
                if meta is None:
                    continue

                key, pkt_time, wire_len, payload_len = meta

                # Handle Idle Timeout if enabled
                if self.idle_timeout is not None and key in self.active_flows:
                    flow = self.active_flows[key]
                    if (pkt_time - flow.end_time) > self.idle_timeout:
                        expired_records.append(flow.to_record())
                        del self.active_flows[key]

                # Update or initialize flow accumulator
                if key not in self.active_flows:
                    self.active_flows[key] = FlowAccumulator(
                        key=key,
                        start_time=pkt_time,
                        end_time=pkt_time,
                    )
                
                self.active_flows[key].add_packet(pkt_time, wire_len, payload_len, pkt)

        # Flush remaining active flows
        for flow in self.active_flows.values():
            expired_records.append(flow.to_record())

        logger.info(
            f"Extraction Complete. Processed {self.total_packets_processed} packets "
            f"({self.total_ip_packets} IP, {self.total_non_ip_packets} non-IP). "
            f"Extracted {len(expired_records)} distinct flows."
        )
        return expired_records

    def export_to_csv(self, output_csv_path: str) -> pd.DataFrame:
        """
        Extracts flows and exports them to a clean CSV file.
        Returns the resulting pandas DataFrame.
        """
        records = self.extract_flows()
        
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)

        if not records:
            logger.warning(f"No IP flows found in {self.pcap_path}. Writing empty CSV structure.")
            df = pd.DataFrame(columns=[
                "flow_id", "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
                "start_time", "end_time", "flow_duration_sec", "packet_count",
                "total_bytes", "total_payload_bytes", "bytes_per_sec", "pkts_per_sec",
                "payload_bytes_per_sec", "iat_mean", "iat_std", "iat_min", "iat_max", "iat_total",
                "pkt_len_mean", "pkt_len_std", "pkt_len_min", "pkt_len_max",
                "payload_len_mean", "payload_len_std", "payload_len_min", "payload_len_max",
                "payload_ratio", "syn_count", "ack_count", "fin_count", "rst_count",
                "psh_count", "urg_count", "cwr_count", "ece_count", "tcp_win_init",
                "tcp_win_mean", "tcp_win_min", "tcp_win_max", "icmp_type", "icmp_code",
                "is_single_packet"
            ])
        else:
            df = pd.DataFrame(records)

        df.to_csv(output_csv_path, index=False)
        logger.info(f"Saved {len(df)} flow metadata records to CSV: {output_csv_path}")
        return df


def ingest_pcap_to_csv(
    pcap_path: str,
    output_csv_path: str,
    unidirectional: bool = True,
    idle_timeout: Optional[float] = None,
) -> pd.DataFrame:
    """
    Convenience function for reading a PCAP file and exporting metadata to CSV.
    """
    extractor = PassiveFlowExtractor(
        pcap_path=pcap_path,
        unidirectional=unidirectional,
        idle_timeout=idle_timeout,
    )
    return extractor.export_to_csv(output_csv_path)


def print_flow_summary(df: pd.DataFrame) -> None:
    """Prints a clean, informative console summary of extracted network flows."""
    if df.empty:
        print("No flows were recorded.")
        return

    print("\n" + "=" * 70)
    print("                AEGIS SENTINEL NETWORK FLOW INGEST SUMMARY")
    print("=" * 70)
    print(f"Total Flows Extracted       : {len(df):,}")
    print(f"Total Aggregated Packets    : {df['packet_count'].sum():,}")
    print(f"Total Bytes Ingested        : {df['total_bytes'].sum():,} bytes ({df['total_bytes'].sum() / (1024*1024):.2f} MB)")
    print(f"Total Payload Bytes         : {df['total_payload_bytes'].sum():,} bytes")
    print(f"Unique Source IPs           : {df['src_ip'].nunique()}")
    print(f"Unique Destination IPs      : {df['dst_ip'].nunique()}")
    
    print("\n--- Protocol Breakdown ---")
    proto_counts = df["protocol"].value_counts()
    for proto, count in proto_counts.items():
        pct = (count / len(df)) * 100
        print(f"  - {proto:<10} : {count:>6} flows ({pct:5.1f}%)")

    print("\n--- Flow Duration & Rate Statistics ---")
    print(f"  - Duration (sec)     : Mean = {df['flow_duration_sec'].mean():.4f}s, Max = {df['flow_duration_sec'].max():.4f}s")
    print(f"  - Bytes/sec Rate     : Mean = {df['bytes_per_sec'].mean():.2f} B/s, Max = {df['bytes_per_sec'].max():.2f} B/s")
    print(f"  - Inter-Arrival (s)  : Mean = {df['iat_mean'].mean():.6f}s, Max = {df['iat_max'].max():.6f}s")
    print(f"  - Packet Length (B)  : Mean = {df['pkt_len_mean'].mean():.2f} B, Max = {df['pkt_len_max'].max()} B")
    
    if "syn_count" in df.columns:
        print(f"  - TCP Flags Observed : SYN={df['syn_count'].sum()}, ACK={df['ack_count'].sum()}, FIN={df['fin_count'].sum()}, RST={df['rst_count'].sum()}")

    print("=" * 70 + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Aegis Sentinel - Passive Unidirectional PCAP Flow Metadata Extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
  python src/ingest.py --pcap data/raw/capture.pcap --output data/processed/flows.csv
  python src/ingest.py -p data/raw/capture.pcap -o data/processed/flows.csv --idle-timeout 60
  python src/ingest.py -p data/raw/capture.pcap -o data/processed/flows.csv --bidirectional
        """,
    )
    parser.add_argument(
        "-p", "--pcap",
        required=True,
        help="Path to input .pcap or .pcapng capture file (read-only)",
    )
    parser.add_argument(
        "-o", "--output",
        default="data/processed/flows.csv",
        help="Path to destination .csv output file (default: data/processed/flows.csv)",
    )
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        default=False,
        help="Enable bidirectional pairing (default: False - strictly unidirectional / data diode mode)",
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=None,
        help="Flow inactivity timeout in seconds (optional)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress summary statistics output",
    )
    return parser


def main() -> None:
    """CLI execution entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()

    unidirectional = not args.bidirectional

    try:
        start_wall_time = time.time()
        df = ingest_pcap_to_csv(
            pcap_path=args.pcap,
            output_csv_path=args.output,
            unidirectional=unidirectional,
            idle_timeout=args.idle_timeout,
        )
        elapsed = time.time() - start_wall_time
        logger.info(f"Ingestion pipeline finished in {elapsed:.3f} seconds.")

        if not args.quiet:
            print_flow_summary(df)

    except Exception as exc:
        logger.error(f"Ingest failed with error: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

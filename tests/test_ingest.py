"""
Unit & Integration Tests for Aegis Sentinel PCAP Ingestion Pipeline
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd
from scapy.all import ARP, Ether, ICMP, IP, Raw, TCP, UDP, wrpcap

from src.ingest import FlowKey, PassiveFlowExtractor, ingest_pcap_to_csv


class TestPCAPIngest(unittest.TestCase):
    """Test suite for passive network flow metadata extraction."""

    def setUp(self):
        """Create a clean temporary directory for synthetic PCAPs and CSV outputs."""
        self.test_dir = tempfile.mkdtemp(prefix="aegis_test_")
        self.pcap_path = os.path.join(self.test_dir, "test_traffic.pcap")
        self.csv_output = os.path.join(self.test_dir, "extracted_flows.csv")

    def tearDown(self):
        """Clean up temporary directory."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def _generate_synthetic_pcap(self) -> str:
        """
        Creates a synthetic PCAP containing:
        1. TCP flow 1: 192.168.1.100:54321 -> 10.0.0.1:80 (3 packets, timestamps 100.0, 100.5, 101.5)
        2. TCP reverse flow: 10.0.0.1:80 -> 192.168.1.100:54321 (2 packets, timestamps 100.1, 101.0)
        3. UDP flow: 192.168.1.100:41234 -> 8.8.8.8:53 (2 packets, timestamps 102.0, 102.2)
        4. ICMP flow: 192.168.1.100 -> 1.1.1.1 (1 packet, timestamp 103.0)
        5. Non-IP packet: ARP request (should be ignored gracefully)
        """
        packets = []

        # 1. Forward TCP flow (A -> B)
        # Pkt 1: SYN
        p1 = Ether() / IP(src="192.168.1.100", dst="10.0.0.1") / TCP(sport=54321, dport=80, flags="S", window=65535)
        p1.time = 100.0
        packets.append(p1)

        # Pkt 2: ACK + Data (100 bytes payload)
        p2 = Ether() / IP(src="192.168.1.100", dst="10.0.0.1") / TCP(sport=54321, dport=80, flags="PA") / Raw(load=b"X" * 100)
        p2.time = 100.5
        packets.append(p2)

        # Pkt 3: FIN + ACK
        p3 = Ether() / IP(src="192.168.1.100", dst="10.0.0.1") / TCP(sport=54321, dport=80, flags="FA")
        p3.time = 101.5
        packets.append(p3)

        # 2. Reverse TCP flow (B -> A)
        # Pkt 4: SYN-ACK
        p4 = Ether() / IP(src="10.0.0.1", dst="192.168.1.100") / TCP(sport=80, dport=54321, flags="SA", window=32768)
        p4.time = 100.1
        packets.append(p4)

        # Pkt 5: ACK
        p5 = Ether() / IP(src="10.0.0.1", dst="192.168.1.100") / TCP(sport=80, dport=54321, flags="A", window=32768)
        p5.time = 101.0
        packets.append(p5)

        # 3. UDP flow (DNS query style)
        p6 = Ether() / IP(src="192.168.1.100", dst="8.8.8.8") / UDP(sport=41234, dport=53) / Raw(load=b"DNSQUERY")
        p6.time = 102.0
        packets.append(p6)

        p7 = Ether() / IP(src="192.168.1.100", dst="8.8.8.8") / UDP(sport=41234, dport=53) / Raw(load=b"DNSQUERY2")
        p7.time = 102.2
        packets.append(p7)

        # 4. ICMP ping
        p8 = Ether() / IP(src="192.168.1.100", dst="1.1.1.1") / ICMP(type=8, code=0)
        p8.time = 103.0
        packets.append(p8)

        # 5. ARP (Non-IP)
        p9 = Ether() / ARP(op=1, psrc="192.168.1.100", pdst="192.168.1.1")
        p9.time = 104.0
        packets.append(p9)

        wrpcap(self.pcap_path, packets)
        return self.pcap_path

    def test_unidirectional_flow_extraction(self):
        """Verify strict unidirectional flow separation and statistical metric accuracy."""
        self._generate_synthetic_pcap()

        extractor = PassiveFlowExtractor(self.pcap_path, unidirectional=True)
        records = extractor.extract_flows()

        # We expect 4 distinct unidirectional IP flows (Forward TCP, Reverse TCP, UDP, ICMP)
        self.assertEqual(len(records), 4)

        df = pd.DataFrame(records)

        # Find Forward TCP flow: 192.168.1.100:54321 -> 10.0.0.1:80
        fwd_tcp = df[(df["src_ip"] == "192.168.1.100") & (df["dst_ip"] == "10.0.0.1") & (df["protocol"] == "TCP")]
        self.assertEqual(len(fwd_tcp), 1)
        row = fwd_tcp.iloc[0]

        # Verify packet count and duration (101.5 - 100.0 = 1.5s)
        self.assertEqual(row["packet_count"], 3)
        self.assertAlmostEqual(row["flow_duration_sec"], 1.5, places=3)
        
        # Verify Inter-Arrival Times:
        # Delays: [100.5 - 100.0 = 0.5, 101.5 - 100.5 = 1.0] -> Mean = 0.75s, Min = 0.5s, Max = 1.0s
        self.assertAlmostEqual(row["iat_mean"], 0.75, places=4)
        self.assertAlmostEqual(row["iat_min"], 0.5, places=4)
        self.assertAlmostEqual(row["iat_max"], 1.0, places=4)
        self.assertAlmostEqual(row["iat_total"], 1.5, places=4)

        # Verify Bytes and Rates
        self.assertGreater(row["total_bytes"], 0)
        self.assertEqual(row["total_payload_bytes"], 100)
        self.assertAlmostEqual(row["bytes_per_sec"], row["total_bytes"] / 1.5, places=2)
        self.assertAlmostEqual(row["pkts_per_sec"], 3 / 1.5, places=2)

        # Verify TCP Flags
        self.assertEqual(row["syn_count"], 1)
        self.assertEqual(row["ack_count"], 2)  # p2 (PA) and p3 (FA)
        self.assertEqual(row["fin_count"], 1)
        self.assertEqual(row["psh_count"], 1)
        self.assertEqual(row["rst_count"], 0)
        self.assertEqual(row["tcp_win_init"], 65535)

        # Verify Reverse TCP flow is separate
        rev_tcp = df[(df["src_ip"] == "10.0.0.1") & (df["dst_ip"] == "192.168.1.100") & (df["protocol"] == "TCP")]
        self.assertEqual(len(rev_tcp), 1)
        rev_row = rev_tcp.iloc[0]
        self.assertEqual(rev_row["packet_count"], 2)
        self.assertAlmostEqual(rev_row["flow_duration_sec"], 0.9, places=3)  # 101.0 - 100.1
        self.assertEqual(rev_row["syn_count"], 1)
        self.assertEqual(rev_row["ack_count"], 2)

        # Verify UDP flow
        udp_flow = df[df["protocol"] == "UDP"].iloc[0]
        self.assertEqual(udp_flow["src_port"], 41234)
        self.assertEqual(udp_flow["dst_port"], 53)
        self.assertEqual(udp_flow["packet_count"], 2)
        self.assertAlmostEqual(udp_flow["iat_mean"], 0.2, places=4)

        # Verify ICMP flow
        icmp_flow = df[df["protocol"] == "ICMP"].iloc[0]
        self.assertEqual(icmp_flow["packet_count"], 1)
        self.assertEqual(icmp_flow["is_single_packet"], 1)
        self.assertEqual(icmp_flow["flow_duration_sec"], 0.0)
        self.assertEqual(icmp_flow["icmp_type"], 8)
        self.assertEqual(icmp_flow["icmp_code"], 0)

    def test_bidirectional_pairing_option(self):
        """Verify that enabling bidirectional mode merges forward and reverse packets into single flow."""
        self._generate_synthetic_pcap()

        extractor = PassiveFlowExtractor(self.pcap_path, unidirectional=False)
        records = extractor.extract_flows()

        df = pd.DataFrame(records)
        # TCP forward (3 pkts) + reverse (2 pkts) should merge into 1 flow of 5 packets
        tcp_flows = df[df["protocol"] == "TCP"]
        self.assertEqual(len(tcp_flows), 1)
        self.assertEqual(tcp_flows.iloc[0]["packet_count"], 5)

    def test_csv_export_pipeline(self):
        """Test end-to-end export to CSV file."""
        self._generate_synthetic_pcap()
        df = ingest_pcap_to_csv(self.pcap_path, self.csv_output, unidirectional=True)

        self.assertTrue(os.path.exists(self.csv_output))
        loaded_df = pd.read_csv(self.csv_output)
        self.assertEqual(len(loaded_df), 4)
        self.assertIn("flow_duration_sec", loaded_df.columns)
        self.assertIn("bytes_per_sec", loaded_df.columns)
        self.assertIn("iat_mean", loaded_df.columns)


if __name__ == "__main__":
    unittest.main()

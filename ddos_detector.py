#!/usr/bin/env python3
"""
DDoS Detection System
====================

Real-time DDoS attack detection using machine learning and network analysis.
Detects SYN floods, UDP floods, HTTP floods, and other attack patterns.

Author: DDoS Detection System
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import statistics

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import psutil
import socket
import struct
import threading
from scapy.all import sniff, IP, TCP, UDP, Raw
from scapy.layers.inet import IP
from scapy.layers.l2 import Ether

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ddos_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class NetworkStats:
    """Network statistics for a time window."""
    timestamp: datetime
    packet_count: int
    byte_count: int
    unique_sources: int
    unique_destinations: int
    syn_count: int
    syn_ack_count: int
    udp_count: int
    http_count: int
    avg_packet_size: float
    packet_rate: float
    connection_attempts: int
    established_connections: int

@dataclass
class AttackAlert:
    """DDoS attack alert."""
    timestamp: datetime
    attack_type: str
    severity: str
    source_ips: List[str]
    target_ips: List[str]
    packet_count: int
    confidence: float
    description: str

class PacketCapture:
    """Real-time packet capture and analysis."""
    
    def __init__(self, interface: str = None):
        self.interface = interface
        self.packets = deque(maxlen=10000)  # Keep last 10k packets
        self.stats_window = deque(maxlen=60)  # 60-second sliding window
        self.running = False
        self.capture_thread = None
        
    def start_capture(self):
        """Start packet capture in background thread."""
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        logger.info("Packet capture started")
    
    def stop_capture(self):
        """Stop packet capture."""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join()
        logger.info("Packet capture stopped")
    
    def _capture_loop(self):
        """Main packet capture loop."""
        try:
            sniff(
                iface=self.interface,
                prn=self._process_packet,
                stop_filter=lambda x: not self.running,
                store=0
            )
        except Exception as e:
            logger.error(f"Packet capture error: {e}")
    
    def _process_packet(self, packet):
        """Process captured packet."""
        try:
            if IP in packet:
                packet_data = {
                    'timestamp': datetime.now(),
                    'src_ip': packet[IP].src,
                    'dst_ip': packet[IP].dst,
                    'protocol': packet[IP].proto,
                    'length': len(packet),
                    'flags': 0,
                    'src_port': 0,
                    'dst_port': 0
                }
                
                if TCP in packet:
                    packet_data['flags'] = packet[TCP].flags
                    packet_data['src_port'] = packet[TCP].sport
                    packet_data['dst_port'] = packet[TCP].dport
                elif UDP in packet:
                    packet_data['src_port'] = packet[UDP].sport
                    packet_data['dst_port'] = packet[UDP].dport
                
                self.packets.append(packet_data)
                self._update_stats()
                
        except Exception as e:
            logger.error(f"Packet processing error: {e}")
    
    def _update_stats(self):
        """Update network statistics."""
        now = datetime.now()
        window_start = now - timedelta(seconds=60)
        
        # Filter packets in time window
        recent_packets = [p for p in self.packets if p['timestamp'] >= window_start]
        
        if not recent_packets:
            return
        
        # Calculate statistics
        stats = NetworkStats(
            timestamp=now,
            packet_count=len(recent_packets),
            byte_count=sum(p['length'] for p in recent_packets),
            unique_sources=len(set(p['src_ip'] for p in recent_packets)),
            unique_destinations=len(set(p['dst_ip'] for p in recent_packets)),
            syn_count=sum(1 for p in recent_packets if p['flags'] & 0x02),
            syn_ack_count=sum(1 for p in recent_packets if (p['flags'] & 0x12) == 0x12),
            udp_count=sum(1 for p in recent_packets if p['protocol'] == 17),
            http_count=sum(1 for p in recent_packets if p['dst_port'] in [80, 443, 8080]),
            avg_packet_size=statistics.mean(p['length'] for p in recent_packets),
            packet_rate=len(recent_packets) / 60.0,
            connection_attempts=sum(1 for p in recent_packets if p['flags'] & 0x02),
            established_connections=sum(1 for p in recent_packets if (p['flags'] & 0x12) == 0x12)
        )
        
        self.stats_window.append(stats)

class DDoSDetector:
    """Main DDoS detection engine."""
    
    def __init__(self):
        self.packet_capture = PacketCapture()
        self.ml_model = None
        self.scaler = StandardScaler()
        self.attack_thresholds = {
            'syn_flood': {'packet_rate': 1000, 'syn_ratio': 0.1},
            'udp_flood': {'packet_rate': 2000, 'udp_ratio': 0.8},
            'http_flood': {'packet_rate': 500, 'http_ratio': 0.9}
        }
        self.alerts = deque(maxlen=1000)
        self.training_data = []
        
    def start_detection(self):
        """Start the DDoS detection system."""
        logger.info("Starting DDoS detection system")
        self.packet_capture.start_capture()
        
        # Start detection loop
        asyncio.create_task(self._detection_loop())
    
    async def _detection_loop(self):
        """Main detection loop."""
        while True:
            try:
                await self._analyze_traffic()
                await asyncio.sleep(1)  # Check every second
            except Exception as e:
                logger.error(f"Detection loop error: {e}")
                await asyncio.sleep(5)
    
    async def _analyze_traffic(self):
        """Analyze current traffic for DDoS patterns."""
        if not self.packet_capture.stats_window:
            return
        
        latest_stats = self.packet_capture.stats_window[-1]
        
        # Check for SYN flood
        if self._detect_syn_flood(latest_stats):
            alert = AttackAlert(
                timestamp=datetime.now(),
                attack_type="SYN Flood",
                severity="High",
                source_ips=self._get_top_sources(),
                target_ips=self._get_top_destinations(),
                packet_count=latest_stats.packet_count,
                confidence=0.85,
                description=f"High SYN packet rate: {latest_stats.packet_rate:.1f} pps"
            )
            self.alerts.append(alert)
            logger.warning(f"SYN Flood detected: {alert.description}")
        
        # Check for UDP flood
        if self._detect_udp_flood(latest_stats):
            alert = AttackAlert(
                timestamp=datetime.now(),
                attack_type="UDP Flood",
                severity="High",
                source_ips=self._get_top_sources(),
                target_ips=self._get_top_destinations(),
                packet_count=latest_stats.packet_count,
                confidence=0.80,
                description=f"High UDP packet rate: {latest_stats.packet_rate:.1f} pps"
            )
            self.alerts.append(alert)
            logger.warning(f"UDP Flood detected: {alert.description}")
        
        # Check for HTTP flood
        if self._detect_http_flood(latest_stats):
            alert = AttackAlert(
                timestamp=datetime.now(),
                attack_type="HTTP Flood",
                severity="Medium",
                source_ips=self._get_top_sources(),
                target_ips=self._get_top_destinations(),
                packet_count=latest_stats.packet_count,
                confidence=0.75,
                description=f"High HTTP traffic: {latest_stats.http_count} requests"
            )
            self.alerts.append(alert)
            logger.warning(f"HTTP Flood detected: {alert.description}")
    
    def _detect_syn_flood(self, stats: NetworkStats) -> bool:
        """Detect SYN flood attack."""
        if stats.packet_rate < self.attack_thresholds['syn_flood']['packet_rate']:
            return False
        
        syn_ratio = stats.syn_count / stats.packet_count if stats.packet_count > 0 else 0
        return syn_ratio > self.attack_thresholds['syn_flood']['syn_ratio']
    
    def _detect_udp_flood(self, stats: NetworkStats) -> bool:
        """Detect UDP flood attack."""
        if stats.packet_rate < self.attack_thresholds['udp_flood']['packet_rate']:
            return False
        
        udp_ratio = stats.udp_count / stats.packet_count if stats.packet_count > 0 else 0
        return udp_ratio > self.attack_thresholds['udp_flood']['udp_ratio']
    
    def _detect_http_flood(self, stats: NetworkStats) -> bool:
        """Detect HTTP flood attack."""
        if stats.packet_rate < self.attack_thresholds['http_flood']['packet_rate']:
            return False
        
        http_ratio = stats.http_count / stats.packet_count if stats.packet_count > 0 else 0
        return http_ratio > self.attack_thresholds['http_flood']['http_ratio']
    
    def _get_top_sources(self, limit: int = 10) -> List[str]:
        """Get top source IPs by packet count."""
        if not self.packet_capture.packets:
            return []
        
        source_counts = defaultdict(int)
        for packet in self.packet_capture.packets:
            source_counts[packet['src_ip']] += 1
        
        return sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
    
    def _get_top_destinations(self, limit: int = 10) -> List[str]:
        """Get top destination IPs by packet count."""
        if not self.packet_capture.packets:
            return []
        
        dest_counts = defaultdict(int)
        for packet in self.packet_capture.packets:
            dest_counts[packet['dst_ip']] += 1
        
        return sorted(dest_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
    
    def train_ml_model(self, training_data: List[Dict]):
        """Train machine learning model for attack detection."""
        if not training_data:
            logger.warning("No training data provided")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(training_data)
        
        # Feature engineering
        features = [
            'packet_count', 'byte_count', 'unique_sources', 'unique_destinations',
            'syn_count', 'syn_ack_count', 'udp_count', 'http_count',
            'avg_packet_size', 'packet_rate', 'connection_attempts', 'established_connections'
        ]
        
        X = df[features]
        y = df['is_attack'] if 'is_attack' in df.columns else np.zeros(len(df))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.ml_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.ml_model.predict(X_test_scaled)
        logger.info(f"Model trained. Accuracy: {self.ml_model.score(X_test_scaled, y_test):.3f}")
        
        return classification_report(y_test, y_pred)
    
    def get_current_stats(self) -> Dict:
        """Get current network statistics."""
        if not self.packet_capture.stats_window:
            return {}
        
        latest = self.packet_capture.stats_window[-1]
        return {
            'timestamp': latest.timestamp.isoformat(),
            'packet_count': latest.packet_count,
            'byte_count': latest.byte_count,
            'packet_rate': latest.packet_rate,
            'unique_sources': latest.unique_sources,
            'unique_destinations': latest.unique_destinations,
            'syn_ratio': latest.syn_count / latest.packet_count if latest.packet_count > 0 else 0,
            'udp_ratio': latest.udp_count / latest.packet_count if latest.packet_count > 0 else 0,
            'http_ratio': latest.http_count / latest.packet_count if latest.packet_count > 0 else 0
        }
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """Get recent attack alerts."""
        return [
            {
                'timestamp': alert.timestamp.isoformat(),
                'attack_type': alert.attack_type,
                'severity': alert.severity,
                'source_ips': alert.source_ips,
                'target_ips': alert.target_ips,
                'packet_count': alert.packet_count,
                'confidence': alert.confidence,
                'description': alert.description
            }
            for alert in list(self.alerts)[-limit:]
        ]

def main():
    """Main function to run DDoS detection."""
    print("🚀 Starting DDoS Detection System")
    print("=" * 50)
    
    # Create detector
    detector = DDoSDetector()
    
    try:
        # Start detection
        detector.start_detection()
        
        print("✅ DDoS detection system is running")
        print("📊 Monitoring network traffic...")
        print("🛑 Press Ctrl+C to stop")
        
        # Keep running
        while True:
            time.sleep(1)
            
            # Print current stats every 10 seconds
            if int(time.time()) % 10 == 0:
                stats = detector.get_current_stats()
                if stats:
                    print(f"\n📈 Current Stats:")
                    print(f"   Packets/sec: {stats.get('packet_rate', 0):.1f}")
                    print(f"   Unique sources: {stats.get('unique_sources', 0)}")
                    print(f"   SYN ratio: {stats.get('syn_ratio', 0):.3f}")
                    print(f"   UDP ratio: {stats.get('udp_ratio', 0):.3f}")
                
                # Check for alerts
                alerts = detector.get_recent_alerts(1)
                if alerts:
                    alert = alerts[0]
                    print(f"🚨 ALERT: {alert['attack_type']} - {alert['description']}")
    
    except KeyboardInterrupt:
        print("\n🛑 Stopping DDoS detection system...")
        detector.packet_capture.stop_capture()
        print("✅ System stopped")

if __name__ == "__main__":
    main()

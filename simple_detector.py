#!/usr/bin/env python3
"""
Simple DDoS Detection System
============================

A simplified version that works without requiring admin privileges.
Uses simulated data for demonstration purposes.

Author: DDoS Detection System
"""

import time
import random
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from collections import deque, Counter, defaultdict
import threading
import math
import argparse
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DetectorConfig:
    window_seconds: int = 60
    check_interval_seconds: int = 5
    simulate_interval_seconds: float = 0.1
    # Adaptive thresholds (z-score)
    rate_z_threshold: float = 4.0
    syn_ratio_threshold: float = 0.8
    udp_ratio_threshold: float = 0.9
    http_ratio_threshold: float = 0.7
    # Targeted flood per-destination thresholds
    per_dst_rate_z_threshold: float = 4.0
    per_dst_min_pps: float = 25.0
    # Alerting
    alert_cooldown_seconds: int = 60
    max_alerts: int = 200
    # Entropy anomaly
    min_src_entropy_bits: float = 6.0
    # Modes
    simulate: bool = True


class SimpleDDoSDetector:
    """Simplified DDoS detection system."""
    
    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()
        self.running = False
        self.stats = {
            'packet_count': 0,
            'packet_rate': 0.0,
            'unique_sources': 0,
            'unique_destinations': 0,
            'syn_ratio': 0.0,
            'udp_ratio': 0.0,
            'http_ratio': 0.0,
            'src_entropy_bits': 0.0,
            'dst_port_entropy_bits': 0.0,
            'timestamp': datetime.now().isoformat()
        }
        self.alerts = deque(maxlen=self.config.max_alerts)
        # Keep enough history for windowed stats; target ~10x typical pps at simulate rate
        max_packets_estimate = max(1000, int(self.config.window_seconds / max(self.config.simulate_interval_seconds, 0.01)) * 20)
        self.packet_history = deque(maxlen=max_packets_estimate)
        # Adaptive baseline (EWMA / variance) for rate
        self._ewma_rate = 0.0
        self._ewma_rate_sq = 0.0
        self._ewma_alpha = 2.0 / (self.config.window_seconds / self.config.check_interval_seconds + 1.0)
        # Per-destination tracking
        self._per_dst_counts: Counter[str] = Counter()
        self._per_dst_ewma_rate: defaultdict[str, float] = defaultdict(float)
        self._per_dst_ewma_rate_sq: defaultdict[str, float] = defaultdict(float)
        # Alert cooldowns
        self._last_alert_at: Dict[Tuple[str, str], datetime] = {}
    
    def start_detection(self):
        """Start the detection system."""
        self.running = True
        logger.info("Starting simple DDoS detection system")
        
        if self.config.simulate:
            simulation_thread = threading.Thread(target=self._simulate_traffic)
            simulation_thread.daemon = True
            simulation_thread.start()
        
        # Start detection thread
        detection_thread = threading.Thread(target=self._detection_loop)
        detection_thread.daemon = True
        detection_thread.start()
    
    def stop_detection(self):
        """Stop the detection system."""
        self.running = False
        logger.info("DDoS detection system stopped")
    
    def _simulate_traffic(self):
        """Simulate network traffic."""
        while self.running:
            try:
                # Generate simulated packet
                packet = {
                    'timestamp': datetime.now(),
                    'src_ip': f"192.168.1.{random.randint(1, 254)}",
                    'dst_ip': f"192.168.1.{random.randint(1, 254)}",
                    'protocol': random.choice([6, 17]),  # TCP or UDP
                    'length': random.randint(40, 1500),
                    'flags': random.choice([0x02, 0x12, 0x10, 0x00]),
                    'src_port': random.randint(1024, 65535),
                    'dst_port': random.choice([80, 443, 22, 53, 8080])
                }
                
                self.packet_history.append(packet)
                self._update_stats()
                
                time.sleep(self.config.simulate_interval_seconds)
                
            except Exception as e:
                logger.error(f"Traffic simulation error: {e}")
                time.sleep(1)
    
    def _update_stats(self):
        """Update network statistics."""
        now = datetime.now()
        window_start = now - timedelta(seconds=self.config.window_seconds)
        
        # Filter recent packets
        recent_packets = [p for p in self.packet_history if p['timestamp'] >= window_start]
        
        if not recent_packets:
            return
        
        # Calculate statistics
        packet_count = len(recent_packets)
        packet_rate = packet_count / float(self.config.window_seconds)
        unique_sources = len(set(p['src_ip'] for p in recent_packets))
        unique_destinations = len(set(p['dst_ip'] for p in recent_packets))
        syn_ratio = sum(1 for p in recent_packets if p['flags'] & 0x02) / packet_count
        udp_ratio = sum(1 for p in recent_packets if p['protocol'] == 17) / packet_count
        http_ratio = sum(1 for p in recent_packets if p['dst_port'] in [80, 443, 8080]) / packet_count

        # Entropy features
        def shannon_entropy(values: List[str]) -> float:
            if not values:
                return 0.0
            counts = Counter(values)
            total = float(len(values))
            return -sum((c / total) * math.log2(c / total) for c in counts.values())

        src_entropy_bits = shannon_entropy([p['src_ip'] for p in recent_packets])
        dst_port_entropy_bits = shannon_entropy([str(p['dst_port']) for p in recent_packets])

        # Update per-destination counts
        self._per_dst_counts.clear()
        for p in recent_packets:
            self._per_dst_counts[p['dst_ip']] += 1

        # Update EWMA baselines (overall rate)
        alpha = self._ewma_alpha
        self._ewma_rate = (1 - alpha) * self._ewma_rate + alpha * packet_rate
        self._ewma_rate_sq = (1 - alpha) * self._ewma_rate_sq + alpha * (packet_rate * packet_rate)

        # Update per-destination EWMA baselines
        for dst_ip, cnt in self._per_dst_counts.items():
            dst_rate = cnt / float(self.config.window_seconds)
            self._per_dst_ewma_rate[dst_ip] = (1 - alpha) * self._per_dst_ewma_rate[dst_ip] + alpha * dst_rate
            self._per_dst_ewma_rate_sq[dst_ip] = (1 - alpha) * self._per_dst_ewma_rate_sq[dst_ip] + alpha * (dst_rate * dst_rate)

        self.stats = {
            'packet_count': packet_count,
            'packet_rate': packet_rate,
            'unique_sources': unique_sources,
            'unique_destinations': unique_destinations,
            'syn_ratio': syn_ratio,
            'udp_ratio': udp_ratio,
            'http_ratio': http_ratio,
            'src_entropy_bits': src_entropy_bits,
            'dst_port_entropy_bits': dst_port_entropy_bits,
            'timestamp': now.isoformat()
        }
    
    def _detection_loop(self):
        """Main detection loop."""
        while self.running:
            try:
                self._check_for_attacks()
                time.sleep(self.config.check_interval_seconds)
            except Exception as e:
                logger.error(f"Detection loop error: {e}")
                time.sleep(self.config.check_interval_seconds)
    
    def _check_for_attacks(self):
        """Check for DDoS attacks."""
        if not self.stats['packet_count']:
            return
        
        # Compute z-score for overall packet_rate
        mean = self._ewma_rate
        var = max(self._ewma_rate_sq - mean * mean, 1e-8)
        std = math.sqrt(var)
        z_rate = (self.stats['packet_rate'] - mean) / std if std > 0 else 0.0

        # Check SYN flood (ratio + anomalous rate)
        if (z_rate >= self.config.rate_z_threshold and self.stats['syn_ratio'] >= self.config.syn_ratio_threshold):
            self._create_alert(
                attack_type="SYN Flood",
                severity="High",
                description=f"Anomalous rate z={z_rate:.1f} with SYN ratio {self.stats['syn_ratio']:.2f}",
                key_details={}
            )

        # Check UDP flood
        if (z_rate >= self.config.rate_z_threshold and self.stats['udp_ratio'] >= self.config.udp_ratio_threshold):
            self._create_alert(
                attack_type="UDP Flood",
                severity="High",
                description=f"Anomalous rate z={z_rate:.1f} with UDP ratio {self.stats['udp_ratio']:.2f}",
                key_details={}
            )

        # Check HTTP flood
        if (z_rate >= self.config.rate_z_threshold and self.stats['http_ratio'] >= self.config.http_ratio_threshold):
            self._create_alert(
                attack_type="HTTP Flood",
                severity="Medium",
                description=f"Anomalous rate z={z_rate:.1f} with HTTP ratio {self.stats['http_ratio']:.2f}",
                key_details={}
            )

        # Low source entropy (many hosts participating) is suspicious under high rate
        if (z_rate >= self.config.rate_z_threshold and self.stats['src_entropy_bits'] >= self.config.min_src_entropy_bits):
            self._create_alert(
                attack_type="Distributed Pattern",
                severity="Medium",
                description=f"High rate z={z_rate:.1f} with high src entropy {self.stats['src_entropy_bits']:.1f} bits",
                key_details={}
            )

        # Targeted per-destination flood detection
        for dst_ip, cnt in list(self._per_dst_counts.items()):
            dst_rate = cnt / float(self.config.window_seconds)
            dst_mean = self._per_dst_ewma_rate[dst_ip]
            dst_var = max(self._per_dst_ewma_rate_sq[dst_ip] - dst_mean * dst_mean, 1e-8)
            dst_std = math.sqrt(dst_var)
            dst_z = (dst_rate - dst_mean) / dst_std if dst_std > 0 else 0.0
            if dst_rate >= self.config.per_dst_min_pps and dst_z >= self.config.per_dst_rate_z_threshold:
                self._create_alert(
                    attack_type="Targeted Flood",
                    severity="High",
                    description=f"Dst {dst_ip} rate {dst_rate:.1f} pps (z={dst_z:.1f})",
                    key_details={"dst_ip": dst_ip}
                )
    
    def _create_alert(self, attack_type: str, severity: str, description: str, key_details: Dict | None = None):
        """Create an attack alert."""
        key_details = key_details or {}
        now = datetime.now()
        cooldown_key = (attack_type, json.dumps(key_details, sort_keys=True))
        last_at = self._last_alert_at.get(cooldown_key)
        if last_at and (now - last_at).total_seconds() < self.config.alert_cooldown_seconds:
            return
        self._last_alert_at[cooldown_key] = now

        # Simple confidence heuristic
        confidence = 0.9 if severity == "High" else 0.75

        alert = {
            'timestamp': now.isoformat(),
            'attack_type': attack_type,
            'severity': severity,
            'description': description,
            'packet_count': self.stats['packet_count'],
            'packet_rate': self.stats['packet_rate'],
            'unique_sources': self.stats['unique_sources'],
            'syn_ratio': self.stats['syn_ratio'],
            'udp_ratio': self.stats['udp_ratio'],
            'http_ratio': self.stats['http_ratio'],
            'src_entropy_bits': self.stats['src_entropy_bits'],
            'details': key_details,
            'confidence': confidence
        }

        self.alerts.append(alert)
        logger.warning(f"🚨 {attack_type} detected: {description}")
    
    def get_current_stats(self) -> Dict:
        """Get current network statistics."""
        return self.stats.copy()
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """Get recent attack alerts."""
        return list(self.alerts)[-limit:]

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple DDoS Detection System (adaptive)")
    parser.add_argument("--window", type=int, default=60, help="Sliding window seconds")
    parser.add_argument("--check-interval", type=int, default=5, help="Detection check interval seconds")
    parser.add_argument("--simulate", action="store_true", help="Enable traffic simulation")
    parser.add_argument("--no-simulate", action="store_true", help="Disable traffic simulation")
    parser.add_argument("--cooldown", type=int, default=60, help="Alert cooldown seconds")
    parser.add_argument("--rate-z", type=float, default=4.0, help="Overall rate z-score threshold")
    parser.add_argument("--per-dst-z", type=float, default=4.0, help="Per-destination rate z-score threshold")
    parser.add_argument("--per-dst-min", type=float, default=25.0, help="Per-destination minimum pps to consider")
    parser.add_argument("--syn-ratio", type=float, default=0.8, help="SYN ratio threshold")
    parser.add_argument("--udp-ratio", type=float, default=0.9, help="UDP ratio threshold")
    parser.add_argument("--http-ratio", type=float, default=0.7, help="HTTP ratio threshold")
    parser.add_argument("--min-src-entropy", type=float, default=6.0, help="Min source entropy bits for distributed alarm")
    return parser


def main():
    """Main function."""
    print("🚀 Starting Simple DDoS Detection System (adaptive)")
    print("=" * 50)

    parser = _build_arg_parser()
    args = parser.parse_args()

    simulate = True
    if args.simulate:
        simulate = True
    if args.no_simulate:
        simulate = False

    config = DetectorConfig(
        window_seconds=args.window,
        check_interval_seconds=args.check_interval,
        simulate=simulate,
        alert_cooldown_seconds=args.cooldown,
        rate_z_threshold=args.rate_z,
        per_dst_rate_z_threshold=args.per_dst_z,
        per_dst_min_pps=args.per_dst_min,
        syn_ratio_threshold=args.syn_ratio,
        udp_ratio_threshold=args.udp_ratio,
        http_ratio_threshold=args.http_ratio,
        min_src_entropy_bits=args.min_src_entropy,
    )

    detector = SimpleDDoSDetector(config)
    
    try:
        detector.start_detection()
        
        print("✅ Detection system is running")
        if detector.config.simulate:
            print("📊 Monitoring simulated network traffic...")
        else:
            print("📊 Monitoring... (data source not configured in this demo)")
        print("🛑 Press Ctrl+C to stop")
        
        # Display stats every 10 seconds
        while True:
            time.sleep(10)
            
            stats = detector.get_current_stats()
            print(f"\n📈 Current Stats:")
            print(f"   Packets/sec: {stats['packet_rate']:.1f}")
            print(f"   Unique sources: {stats['unique_sources']}")
            print(f"   SYN ratio: {stats['syn_ratio']:.3f}")
            print(f"   UDP ratio: {stats['udp_ratio']:.3f}")
            print(f"   Src entropy (bits): {stats['src_entropy_bits']:.2f}")
            
            # Check for alerts
            alerts = detector.get_recent_alerts(1)
            if alerts:
                alert = alerts[0]
                print(f"🚨 ALERT: {alert['attack_type']} - {alert['description']}")
    
    except KeyboardInterrupt:
        print("\n🛑 Stopping detection system...")
        detector.stop_detection()
        print("✅ System stopped")

if __name__ == "__main__":
    main()

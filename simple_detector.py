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
from typing import Dict, List
from collections import deque
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleDDoSDetector:
    """Simplified DDoS detection system."""
    
    def __init__(self):
        self.running = False
        self.stats = {
            'packet_count': 0,
            'packet_rate': 0.0,
            'unique_sources': 0,
            'unique_destinations': 0,
            'syn_ratio': 0.0,
            'udp_ratio': 0.0,
            'http_ratio': 0.0,
            'timestamp': datetime.now().isoformat()
        }
        self.alerts = deque(maxlen=100)
        self.packet_history = deque(maxlen=1000)
        self.attack_thresholds = {
            'syn_flood': {'packet_rate': 100, 'syn_ratio': 0.8},
            'udp_flood': {'packet_rate': 200, 'udp_ratio': 0.9},
            'http_flood': {'packet_rate': 50, 'http_ratio': 0.7}
        }
    
    def start_detection(self):
        """Start the detection system."""
        self.running = True
        logger.info("Starting simple DDoS detection system")
        
        # Start simulation thread
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
                
                time.sleep(0.1)  # Simulate packet rate
                
            except Exception as e:
                logger.error(f"Traffic simulation error: {e}")
                time.sleep(1)
    
    def _update_stats(self):
        """Update network statistics."""
        now = datetime.now()
        window_start = now - timedelta(seconds=60)
        
        # Filter recent packets
        recent_packets = [p for p in self.packet_history if p['timestamp'] >= window_start]
        
        if not recent_packets:
            return
        
        # Calculate statistics
        self.stats = {
            'packet_count': len(recent_packets),
            'packet_rate': len(recent_packets) / 60.0,
            'unique_sources': len(set(p['src_ip'] for p in recent_packets)),
            'unique_destinations': len(set(p['dst_ip'] for p in recent_packets)),
            'syn_ratio': sum(1 for p in recent_packets if p['flags'] & 0x02) / len(recent_packets),
            'udp_ratio': sum(1 for p in recent_packets if p['protocol'] == 17) / len(recent_packets),
            'http_ratio': sum(1 for p in recent_packets if p['dst_port'] in [80, 443, 8080]) / len(recent_packets),
            'timestamp': now.isoformat()
        }
    
    def _detection_loop(self):
        """Main detection loop."""
        while self.running:
            try:
                self._check_for_attacks()
                time.sleep(5)  # Check every 5 seconds
            except Exception as e:
                logger.error(f"Detection loop error: {e}")
                time.sleep(5)
    
    def _check_for_attacks(self):
        """Check for DDoS attacks."""
        if not self.stats['packet_count']:
            return
        
        # Check SYN flood
        if (self.stats['packet_rate'] > self.attack_thresholds['syn_flood']['packet_rate'] and
            self.stats['syn_ratio'] > self.attack_thresholds['syn_flood']['syn_ratio']):
            self._create_alert("SYN Flood", "High", f"High SYN rate: {self.stats['packet_rate']:.1f} pps")
        
        # Check UDP flood
        if (self.stats['packet_rate'] > self.attack_thresholds['udp_flood']['packet_rate'] and
            self.stats['udp_ratio'] > self.attack_thresholds['udp_flood']['udp_ratio']):
            self._create_alert("UDP Flood", "High", f"High UDP rate: {self.stats['packet_rate']:.1f} pps")
        
        # Check HTTP flood
        if (self.stats['packet_rate'] > self.attack_thresholds['http_flood']['packet_rate'] and
            self.stats['http_ratio'] > self.attack_thresholds['http_flood']['http_ratio']):
            self._create_alert("HTTP Flood", "Medium", f"High HTTP rate: {self.stats['packet_rate']:.1f} pps")
    
    def _create_alert(self, attack_type: str, severity: str, description: str):
        """Create an attack alert."""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'attack_type': attack_type,
            'severity': severity,
            'description': description,
            'packet_count': self.stats['packet_count'],
            'confidence': 0.85
        }
        
        self.alerts.append(alert)
        logger.warning(f"🚨 {attack_type} detected: {description}")
    
    def get_current_stats(self) -> Dict:
        """Get current network statistics."""
        return self.stats.copy()
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """Get recent attack alerts."""
        return list(self.alerts)[-limit:]

def main():
    """Main function."""
    print("🚀 Starting Simple DDoS Detection System")
    print("=" * 50)
    
    detector = SimpleDDoSDetector()
    
    try:
        detector.start_detection()
        
        print("✅ Detection system is running")
        print("📊 Monitoring simulated network traffic...")
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

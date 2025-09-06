#!/usr/bin/env python3
"""
Test the DDoS Detection System
=============================

Simple test script to verify the system works.
"""

import sys
import time
from simple_detector import SimpleDDoSDetector

def test_detector():
    """Test the simple detector."""
    print("🧪 Testing Simple DDoS Detector")
    print("=" * 40)
    
    # Create detector
    detector = SimpleDDoSDetector()
    
    # Start detection
    detector.start_detection()
    
    print("✅ Detector started successfully")
    
    # Test for 10 seconds
    for i in range(10):
        time.sleep(1)
        
        stats = detector.get_current_stats()
        alerts = detector.get_recent_alerts(1)
        
        print(f"Second {i+1}: Packets/sec: {stats['packet_rate']:.1f}, Sources: {stats['unique_sources']}")
        
        if alerts:
            print(f"🚨 Alert: {alerts[0]['attack_type']}")
    
    # Stop detector
    detector.stop_detection()
    print("✅ Test completed successfully")

if __name__ == "__main__":
    test_detector()

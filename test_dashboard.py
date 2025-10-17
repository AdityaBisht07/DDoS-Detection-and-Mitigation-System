#!/usr/bin/env python3
"""
Test Dashboard
==============

Simple test to verify the dashboard works correctly.
"""

import requests
import time
import subprocess
import sys
import threading

def test_dashboard():
    """Test if the dashboard is working."""
    print("🧪 Testing Dashboard")
    print("=" * 30)
    
    # Start dashboard in background
    process = subprocess.Popen([
        sys.executable, 'simple_dashboard.py'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for startup
    print("⏳ Waiting for dashboard to start...")
    time.sleep(8)
    
    try:
        # Test API endpoints
        print("📡 Testing API endpoints...")
        
        # Test stats endpoint
        response = requests.get('http://localhost:8001/api/stats', timeout=5)
        if response.status_code == 200:
            print("✅ Stats API working")
            stats = response.json()
            print(f"   Packets/sec: {stats.get('packet_rate', 0):.1f}")
        else:
            print(f"❌ Stats API failed: {response.status_code}")
        
        # Test alerts endpoint
        response = requests.get('http://localhost:8001/api/alerts', timeout=5)
        if response.status_code == 200:
            print("✅ Alerts API working")
        else:
            print(f"❌ Alerts API failed: {response.status_code}")
        
        # Test main page
        response = requests.get('http://localhost:8001/', timeout=5)
        if response.status_code == 200:
            print("✅ Dashboard page working")
        else:
            print(f"❌ Dashboard page failed: {response.status_code}")
        
        print("\n🎉 Dashboard test completed successfully!")
        print("🌐 Open http://localhost:8001 in your browser to see the dashboard")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Dashboard test failed: {e}")
        print("Make sure the dashboard is running on port 8001")
    
    finally:
        # Stop the process
        process.terminate()
        process.wait()
        print("🛑 Dashboard stopped")

if __name__ == "__main__":
    test_dashboard()

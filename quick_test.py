#!/usr/bin/env python3
"""
Quick Test for DDoS Detection System
===================================

This script quickly tests if the system components work.
"""

import sys
import time
import subprocess
import threading

def test_imports():
    """Test if all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        import fastapi
        import uvicorn
        import pandas
        import numpy
        import sklearn
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_detector():
    """Test the simple detector."""
    print("🧪 Testing detector...")
    
    try:
        from simple_detector import SimpleDDoSDetector
        
        detector = SimpleDDoSDetector()
        detector.start_detection()
        
        # Test for 3 seconds
        time.sleep(3)
        
        stats = detector.get_current_stats()
        print(f"✅ Detector working - Packets/sec: {stats['packet_rate']:.1f}")
        
        detector.stop_detection()
        return True
        
    except Exception as e:
        print(f"❌ Detector error: {e}")
        return False

def test_dashboard():
    """Test the dashboard startup."""
    print("🧪 Testing dashboard...")
    
    try:
        # Start dashboard in background
        process = subprocess.Popen([
            sys.executable, 'simple_dashboard.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for startup
        time.sleep(5)
        
        if process.poll() is None:
            print("✅ Dashboard started successfully")
            process.terminate()
            return True
        else:
            print("❌ Dashboard failed to start")
            return False
            
    except Exception as e:
        print(f"❌ Dashboard error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Quick Test for DDoS Detection System")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Detector", test_detector),
        ("Dashboard", test_dashboard)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- Testing {test_name} ---")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} test failed")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to run.")
        print("Run: python run_fixed_system.py")
    else:
        print("⚠️ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()

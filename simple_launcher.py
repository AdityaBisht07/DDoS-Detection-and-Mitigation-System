#!/usr/bin/env python3
"""
Simple DDoS Detection System Launcher
=====================================

A simplified launcher that works without admin privileges.
Uses simulated data for demonstration.

Author: DDoS Detection System
"""

import subprocess
import sys
import time
import signal
import os
import threading
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleDDoSSystem:
    """Simple system launcher and manager."""
    
    def __init__(self):
        self.processes = {}
        self.running = False
        
    def check_dependencies(self):
        """Check if all required dependencies are installed."""
        logger.info("Checking dependencies...")
        
        required_packages = [
            'fastapi', 'uvicorn', 'pandas', 'numpy', 'scikit-learn'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing packages: {', '.join(missing_packages)}")
            logger.info("Installing missing packages...")
            return self.install_dependencies()
        
        logger.info("✅ All dependencies are installed")
        return True
    
    def install_dependencies(self):
        """Install required dependencies."""
        logger.info("Installing dependencies...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'fastapi', 'uvicorn', 'pandas', 'numpy', 'scikit-learn'], 
                         check=True, capture_output=True)
            logger.info("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def start_web_dashboard(self):
        """Start the web dashboard."""
        logger.info("Starting web dashboard...")
        try:
            process = subprocess.Popen([
                sys.executable, 'simple_dashboard.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes['web_dashboard'] = process
            logger.info("✅ Web dashboard started on http://localhost:8001")
            return True
        except Exception as e:
            logger.error(f"Failed to start web dashboard: {e}")
            return False
    
    def start_detection_engine(self):
        """Start the detection engine."""
        logger.info("Starting detection engine...")
        try:
            process = subprocess.Popen([
                sys.executable, 'simple_detector.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes['detection_engine'] = process
            logger.info("✅ Detection engine started")
            return True
        except Exception as e:
            logger.error(f"Failed to start detection engine: {e}")
            return False
    
    def start_system(self):
        """Start the complete DDoS detection system."""
        logger.info("🚀 Starting Simple DDoS Detection System")
        logger.info("=" * 50)
        
        # Check dependencies
        if not self.check_dependencies():
            logger.error("❌ Failed to install dependencies")
            return False
        
        # Start components
        if not self.start_web_dashboard():
            logger.error("❌ Failed to start web dashboard")
            return False
        
        # Give web dashboard time to start
        time.sleep(3)
        
        if not self.start_detection_engine():
            logger.error("❌ Failed to start detection engine")
            self.stop_system()
            return False
        
        self.running = True
        
        logger.info("✅ Simple DDoS Detection System is running!")
        logger.info("📊 Dashboard: http://localhost:8001")
        logger.info("📡 API: http://localhost:8001/api/stats")
        logger.info("🔌 WebSocket: ws://localhost:8001/ws")
        logger.info("🛑 Press Ctrl+C to stop")
        
        return True
    
    def stop_system(self):
        """Stop all system components."""
        logger.info("🛑 Stopping Simple DDoS Detection System...")
        
        for name, process in self.processes.items():
            try:
                process.terminate()
                process.wait(timeout=5)
                logger.info(f"✅ {name} stopped")
            except subprocess.TimeoutExpired:
                process.kill()
                logger.warning(f"⚠️ {name} force killed")
            except Exception as e:
                logger.error(f"❌ Error stopping {name}: {e}")
        
        self.processes.clear()
        self.running = False
        logger.info("✅ System stopped")
    
    def monitor_system(self):
        """Monitor system health."""
        while self.running:
            try:
                # Check if processes are still running
                for name, process in list(self.processes.items()):
                    if process.poll() is not None:
                        logger.error(f"❌ {name} has stopped unexpectedly")
                        self.running = False
                        break
                
                time.sleep(5)
                
            except KeyboardInterrupt:
                logger.info("Received interrupt signal")
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(5)
    
    def run(self):
        """Main run method."""
        try:
            if self.start_system():
                self.monitor_system()
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            self.stop_system()

def signal_handler(signum, frame):
    """Handle interrupt signals."""
    logger.info("Received signal, shutting down...")
    sys.exit(0)

def main():
    """Main function."""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🛡️ Simple DDoS Detection System")
    print("=" * 40)
    print("This is a simplified version that works without admin privileges.")
    print("It uses simulated network data for demonstration purposes.")
    print()
    
    # Create and run system
    system = SimpleDDoSSystem()
    system.run()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Fixed DDoS Detection System Launcher
====================================

This launcher runs the working simple version of the DDoS detection system.
No admin privileges required - uses simulated data.

Author: DDoS Detection System
"""

import subprocess
import sys
import time
import signal
import os
import threading
import logging
import webbrowser
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FixedDDoSSystem:
    """Fixed system launcher."""
    
    def __init__(self):
        self.processes = {}
        self.running = False
        
    def check_dependencies(self):
        """Check and install dependencies."""
        logger.info("Checking dependencies...")
        
        try:
            import fastapi
            import uvicorn
            import pandas
            import numpy
            import sklearn
            logger.info("✅ All dependencies are available")
            return True
        except ImportError as e:
            logger.info(f"Installing missing dependencies: {e}")
            return self.install_dependencies()
    
    def install_dependencies(self):
        """Install required dependencies."""
        try:
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', 
                'fastapi', 'uvicorn', 'pandas', 'numpy', 'scikit-learn'
            ], check=True, capture_output=True)
            logger.info("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def start_dashboard(self):
        """Start the web dashboard."""
        logger.info("Starting web dashboard...")
        try:
            # Start dashboard in background
            process = subprocess.Popen([
                sys.executable, 'simple_dashboard.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes['dashboard'] = process
            
            # Wait a moment for startup
            time.sleep(3)
            
            # Check if it's running
            if process.poll() is None:
                logger.info("✅ Web dashboard started successfully")
                return True
            else:
                logger.error("❌ Web dashboard failed to start")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start dashboard: {e}")
            return False
    
    def start_detector(self):
        """Start the detection engine."""
        logger.info("Starting detection engine...")
        try:
            # Start detector in background
            process = subprocess.Popen([
                sys.executable, 'simple_detector.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes['detector'] = process
            
            # Wait a moment for startup
            time.sleep(2)
            
            # Check if it's running
            if process.poll() is None:
                logger.info("✅ Detection engine started successfully")
                return True
            else:
                logger.error("❌ Detection engine failed to start")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start detector: {e}")
            return False
    
    def open_browser(self):
        """Open the dashboard in browser."""
        try:
            webbrowser.open('http://localhost:8001')
            logger.info("🌐 Opening dashboard in browser...")
        except Exception as e:
            logger.warning(f"Could not open browser automatically: {e}")
            logger.info("Please open http://localhost:8001 in your browser")
    
    def start_system(self):
        """Start the complete system."""
        logger.info("🚀 Starting Fixed DDoS Detection System")
        logger.info("=" * 50)
        
        # Check dependencies
        if not self.check_dependencies():
            logger.error("❌ Failed to install dependencies")
            return False
        
        # Start dashboard
        if not self.start_dashboard():
            logger.error("❌ Failed to start dashboard")
            return False
        
        # Start detector
        if not self.start_detector():
            logger.error("❌ Failed to start detector")
            self.stop_system()
            return False
        
        self.running = True
        
        # Open browser
        self.open_browser()
        
        logger.info("✅ System is running successfully!")
        logger.info("📊 Dashboard: http://localhost:8001")
        logger.info("📡 API: http://localhost:8001/api/stats")
        logger.info("🔌 WebSocket: ws://localhost:8001/ws")
        logger.info("🛑 Press Ctrl+C to stop")
        
        return True
    
    def stop_system(self):
        """Stop all components."""
        logger.info("🛑 Stopping system...")
        
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
        try:
            while self.running:
                # Check if processes are still running
                for name, process in list(self.processes.items()):
                    if process.poll() is not None:
                        logger.error(f"❌ {name} has stopped unexpectedly")
                        self.running = False
                        break
                
                time.sleep(5)
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
    
    def run(self):
        """Main run method."""
        try:
            if self.start_system():
                self.monitor_system()
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            self.stop_system()

def main():
    """Main function."""
    print("🛡️ Fixed DDoS Detection System")
    print("=" * 40)
    print("This version uses simulated network data and works without admin privileges.")
    print("Perfect for testing and demonstration purposes.")
    print()
    
    # Create and run system
    system = FixedDDoSSystem()
    system.run()

if __name__ == "__main__":
    main()

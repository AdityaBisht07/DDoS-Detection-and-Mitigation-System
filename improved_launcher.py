#!/usr/bin/env python3
"""
Improved DDoS Detection System Launcher
=======================================

A robust launcher that handles all the issues and provides a working system.
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
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImprovedDDoSSystem:
    """Improved system launcher with better error handling."""
    
    def __init__(self):
        self.processes = {}
        self.running = False
        self.dashboard_port = 8001
        
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
            
            # Wait for startup
            time.sleep(5)
            
            # Check if it's running
            if process.poll() is None:
                logger.info("✅ Web dashboard started successfully")
                return True
            else:
                stdout, stderr = process.communicate()
                logger.error(f"❌ Web dashboard failed to start")
                logger.error(f"STDOUT: {stdout.decode()}")
                logger.error(f"STDERR: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start dashboard: {e}")
            return False
    
    def open_browser(self):
        """Open the dashboard in browser."""
        try:
            webbrowser.open(f'http://localhost:{self.dashboard_port}')
            logger.info("🌐 Opening dashboard in browser...")
        except Exception as e:
            logger.warning(f"Could not open browser automatically: {e}")
            logger.info(f"Please open http://localhost:{self.dashboard_port} in your browser")
    
    def start_system(self):
        """Start the complete system."""
        logger.info("🚀 Starting Improved DDoS Detection System")
        logger.info("=" * 50)
        
        # Check dependencies
        if not self.check_dependencies():
            logger.error("❌ Failed to install dependencies")
            return False
        
        # Start dashboard
        if not self.start_dashboard():
            logger.error("❌ Failed to start dashboard")
            return False
        
        self.running = True
        
        # Open browser
        self.open_browser()
        
        logger.info("✅ System is running successfully!")
        logger.info(f"📊 Dashboard: http://localhost:{self.dashboard_port}")
        logger.info(f"📡 API: http://localhost:{self.dashboard_port}/api/stats")
        logger.info(f"🔌 WebSocket: ws://localhost:{self.dashboard_port}/ws")
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

def signal_handler(signum, frame):
    """Handle interrupt signals."""
    logger.info("Received signal, shutting down...")
    sys.exit(0)

def main():
    """Main function."""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🛡️ Improved DDoS Detection System")
    print("=" * 40)
    print("This version uses simulated network data and works without admin privileges.")
    print("Perfect for testing, demonstration, and learning purposes.")
    print()
    
    # Create and run system
    system = ImprovedDDoSSystem()
    system.run()

if __name__ == "__main__":
    main()

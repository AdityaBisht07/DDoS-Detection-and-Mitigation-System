#!/usr/bin/env python3
"""
Final DDoS Detection System Launcher
====================================

The ultimate launcher that handles all issues and provides multiple options.
Works without admin privileges and includes all necessary components.

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

class FinalDDoSSystem:
    """Final system launcher with all improvements."""
    
    def __init__(self):
        self.processes = {}
        self.running = False
        self.port = 8002
        
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
    
    def start_standalone_dashboard(self):
        """Start the standalone dashboard."""
        logger.info("Starting standalone dashboard...")
        try:
            # Start standalone dashboard in background
            process = subprocess.Popen([
                sys.executable, 'standalone_dashboard.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes['dashboard'] = process
            
            # Wait for startup
            time.sleep(8)
            
            # Check if it's running
            if process.poll() is None:
                logger.info("✅ Standalone dashboard started successfully")
                return True
            else:
                stdout, stderr = process.communicate()
                logger.error(f"❌ Standalone dashboard failed to start")
                logger.error(f"STDOUT: {stdout.decode()}")
                logger.error(f"STDERR: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start standalone dashboard: {e}")
            return False
    
    def open_browser(self):
        """Open the dashboard in browser."""
        try:
            webbrowser.open(f'http://localhost:{self.port}')
            logger.info("🌐 Opening dashboard in browser...")
        except Exception as e:
            logger.warning(f"Could not open browser automatically: {e}")
            logger.info(f"Please open http://localhost:{self.port} in your browser")
    
    def start_system(self):
        """Start the complete system."""
        logger.info("Starting Final DDoS Detection System")
        logger.info("=" * 50)
        
        # Check dependencies
        if not self.check_dependencies():
            logger.error("❌ Failed to install dependencies")
            return False
        
        # Start standalone dashboard
        if not self.start_standalone_dashboard():
            logger.error("❌ Failed to start standalone dashboard")
            return False
        
        self.running = True
        
        # Open browser
        self.open_browser()
        
        logger.info("System is running successfully!")
        logger.info(f"Dashboard: http://localhost:{self.port}")
        logger.info(f"API: http://localhost:{self.port}/api/stats")
        logger.info(f"WebSocket: ws://localhost:{self.port}/ws")
        logger.info("Press Ctrl+C to stop")
        
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
    
    print("Final DDoS Detection System")
    print("=" * 40)
    print("This is the ultimate version that includes everything in one process.")
    print("No admin privileges required - perfect for testing and demonstration.")
    print()
    
    # Create and run system
    system = FinalDDoSSystem()
    system.run()

if __name__ == "__main__":
    main()

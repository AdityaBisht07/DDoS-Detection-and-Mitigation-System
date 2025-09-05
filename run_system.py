#!/usr/bin/env python3
"""
DDoS Detection System Launcher
==============================

Main launcher script that starts all components of the DDoS detection system.
Includes the detection engine, web dashboard, and ML models.

Author: DDoS Detection System
"""

import asyncio
import subprocess
import sys
import time
import signal
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DDoSDetectionSystem:
    """Main system launcher and manager."""
    
    def __init__(self):
        self.processes = {}
        self.running = False
        
    def check_dependencies(self):
        """Check if all required dependencies are installed."""
        logger.info("Checking dependencies...")
        
        required_packages = [
            'fastapi', 'uvicorn', 'pandas', 'numpy', 'scikit-learn', 
            'scapy', 'psutil', 'websockets', 'tensorflow'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing packages: {', '.join(missing_packages)}")
            logger.info("Install missing packages with: pip install -r requirements.txt")
            return False
        
        logger.info("✅ All dependencies are installed")
        return True
    
    def install_dependencies(self):
        """Install required dependencies."""
        logger.info("Installing dependencies...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                         check=True, capture_output=True)
            logger.info("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def train_ml_models(self):
        """Train ML models if they don't exist."""
        models_dir = Path("models")
        if not models_dir.exists() or not any(models_dir.iterdir()):
            logger.info("Training ML models...")
            try:
                subprocess.run([sys.executable, 'ml_trainer.py'], check=True)
                logger.info("✅ ML models trained successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to train ML models: {e}")
                return False
        else:
            logger.info("✅ ML models already exist")
        
        return True
    
    def start_web_dashboard(self):
        """Start the web dashboard."""
        logger.info("Starting web dashboard...")
        try:
            process = subprocess.Popen([
                sys.executable, 'web_dashboard.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes['web_dashboard'] = process
            logger.info("✅ Web dashboard started on http://localhost:8000")
            return True
        except Exception as e:
            logger.error(f"Failed to start web dashboard: {e}")
            return False
    
    def start_detection_engine(self):
        """Start the detection engine."""
        logger.info("Starting detection engine...")
        try:
            process = subprocess.Popen([
                sys.executable, 'ddos_detector.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes['detection_engine'] = process
            logger.info("✅ Detection engine started")
            return True
        except Exception as e:
            logger.error(f"Failed to start detection engine: {e}")
            return False
    
    def start_system(self):
        """Start the complete DDoS detection system."""
        logger.info("🚀 Starting DDoS Detection System")
        logger.info("=" * 50)
        
        # Check dependencies
        if not self.check_dependencies():
            logger.info("Installing missing dependencies...")
            if not self.install_dependencies():
                logger.error("❌ Failed to install dependencies")
                return False
        
        # Train ML models
        if not self.train_ml_models():
            logger.error("❌ Failed to train ML models")
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
        
        logger.info("✅ DDoS Detection System is running!")
        logger.info("📊 Dashboard: http://localhost:8000")
        logger.info("📡 API: http://localhost:8000/api/stats")
        logger.info("🔌 WebSocket: ws://localhost:8000/ws")
        logger.info("🛑 Press Ctrl+C to stop")
        
        return True
    
    def stop_system(self):
        """Stop all system components."""
        logger.info("🛑 Stopping DDoS Detection System...")
        
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
    
    # Check if running as administrator (required for packet capture)
    if os.name == 'nt':  # Windows
        try:
            import ctypes
            if not ctypes.windll.shell32.IsUserAnAdmin():
                print("⚠️  Warning: Running without administrator privileges.")
                print("   Some features may not work properly.")
                print("   For full functionality, run as administrator.")
                print()
        except:
            pass
    
    # Create and run system
    system = DDoSDetectionSystem()
    system.run()

if __name__ == "__main__":
    main()

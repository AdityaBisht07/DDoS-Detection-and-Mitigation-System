#!/usr/bin/env python3
"""
Standalone DDoS Detection Dashboard
===================================

A standalone version that includes both the detector and dashboard in one process.
No separate processes needed - everything runs in one application.

Author: DDoS Detection System
"""

import asyncio
import json
import time
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List
from collections import deque
import threading

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StandaloneDetector:
    """Standalone detector that runs in the same process."""
    
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
        self.simulation_thread = None
        self.detection_thread = None
    
    def start_detection(self):
        """Start the detection system."""
        if self.running:
            return
        
        self.running = True
        logger.info("Starting standalone DDoS detection system")
        
        # Start simulation thread
        self.simulation_thread = threading.Thread(target=self._simulate_traffic)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self._detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
    
    def stop_detection(self):
        """Stop the detection system."""
        self.running = False
        logger.info("Standalone DDoS detection system stopped")
    
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

# Global detector instance
detector = StandaloneDetector()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI."""
    # Startup
    detector.start_detection()
    logger.info("Standalone DDoS detection system started")
    yield
    # Shutdown
    detector.stop_detection()
    logger.info("Standalone DDoS detection system stopped")

# FastAPI app
app = FastAPI(
    title="Standalone DDoS Detection Dashboard", 
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        if not self.active_connections:
            return
        
        message_str = json.dumps(message)
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_str)
            except:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

manager = ConnectionManager()

# Pydantic models
class DetectionConfig(BaseModel):
    syn_flood_threshold: int = 100
    udp_flood_threshold: int = 200
    http_flood_threshold: int = 50
    syn_ratio_threshold: float = 0.8
    udp_ratio_threshold: float = 0.9
    http_ratio_threshold: float = 0.7

# Routes
@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve the main dashboard HTML."""
    return HTMLResponse(content=get_dashboard_html(), status_code=200)

@app.get("/api/stats")
async def get_stats():
    """Get current network statistics."""
    try:
        stats = detector.get_current_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/alerts")
async def get_alerts(limit: int = 50):
    """Get recent attack alerts."""
    try:
        alerts = detector.get_recent_alerts(limit)
        return JSONResponse(content=alerts)
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/config")
async def update_config(config: DetectionConfig):
    """Update detection configuration."""
    try:
        detector.attack_thresholds = {
            'syn_flood': {
                'packet_rate': config.syn_flood_threshold,
                'syn_ratio': config.syn_ratio_threshold
            },
            'udp_flood': {
                'packet_rate': config.udp_flood_threshold,
                'udp_ratio': config.udp_ratio_threshold
            },
            'http_flood': {
                'packet_rate': config.http_flood_threshold,
                'http_ratio': config.http_ratio_threshold
            }
        }
        return JSONResponse(content={"status": "success", "message": "Configuration updated"})
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Send periodic updates
            await asyncio.sleep(2)
            
            # Get current stats
            stats = detector.get_current_stats()
            if stats:
                await websocket.send_text(json.dumps({
                    "type": "stats",
                    "data": stats
                }))
            
            # Check for new alerts
            alerts = detector.get_recent_alerts(1)
            if alerts:
                await websocket.send_text(json.dumps({
                    "type": "alert",
                    "data": alerts[0]
                }))
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

def get_dashboard_html() -> str:
    """Generate the dashboard HTML."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Standalone DDoS Detection Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                min-height: 100vh;
            }
            .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
            .header { text-align: center; margin-bottom: 30px; }
            .header h1 { font-size: 2.5em; margin-bottom: 10px; }
            .header p { font-size: 1.2em; opacity: 0.8; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
            .card { 
                background: rgba(255, 255, 255, 0.1); 
                border-radius: 15px; 
                padding: 20px; 
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            .card h3 { margin-bottom: 15px; font-size: 1.3em; }
            .stat { display: flex; justify-content: space-between; margin: 10px 0; }
            .stat-value { font-weight: bold; font-size: 1.2em; }
            .alert { 
                background: rgba(255, 0, 0, 0.2); 
                border-left: 4px solid #ff4444; 
                padding: 15px; 
                margin: 10px 0;
                border-radius: 5px;
            }
            .alert.high { border-left-color: #ff4444; }
            .alert.medium { border-left-color: #ffaa00; }
            .alert.low { border-left-color: #44ff44; }
            .status { 
                display: inline-block; 
                padding: 5px 15px; 
                border-radius: 20px; 
                font-size: 0.9em;
                font-weight: bold;
            }
            .status.running { background: #44ff44; color: #000; }
            .status.stopped { background: #ff4444; color: #fff; }
            .chart-container { position: relative; height: 300px; margin-top: 20px; }
            .controls { margin-top: 20px; }
            .btn { 
                background: rgba(255, 255, 255, 0.2); 
                border: none; 
                color: white; 
                padding: 10px 20px; 
                border-radius: 5px; 
                cursor: pointer; 
                margin: 5px;
            }
            .btn:hover { background: rgba(255, 255, 255, 0.3); }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🛡️ Standalone DDoS Detection Dashboard</h1>
                <p>Real-time network monitoring and attack detection (All-in-One)</p>
                <div class="status running" id="status">RUNNING</div>
            </div>
            
            <div class="grid">
                <div class="card">
                    <h3>📊 Network Statistics</h3>
                    <div class="stat">
                        <span>Packets/sec:</span>
                        <span class="stat-value" id="packet-rate">0</span>
                    </div>
                    <div class="stat">
                        <span>Total Packets:</span>
                        <span class="stat-value" id="packet-count">0</span>
                    </div>
                    <div class="stat">
                        <span>Unique Sources:</span>
                        <span class="stat-value" id="unique-sources">0</span>
                    </div>
                    <div class="stat">
                        <span>SYN Ratio:</span>
                        <span class="stat-value" id="syn-ratio">0</span>
                    </div>
                    <div class="stat">
                        <span>UDP Ratio:</span>
                        <span class="stat-value" id="udp-ratio">0</span>
                    </div>
                </div>
                
                <div class="card">
                    <h3>🚨 Recent Alerts</h3>
                    <div id="alerts-container">
                        <p>No alerts detected</p>
                    </div>
                </div>
                
                <div class="card">
                    <h3>📈 Traffic Chart</h3>
                    <div class="chart-container">
                        <canvas id="traffic-chart"></canvas>
                    </div>
                </div>
                
                <div class="card">
                    <h3>⚙️ Controls</h3>
                    <div class="controls">
                        <button class="btn" onclick="startDetection()">Start Detection</button>
                        <button class="btn" onclick="stopDetection()">Stop Detection</button>
                        <button class="btn" onclick="clearAlerts()">Clear Alerts</button>
                        <button class="btn" onclick="exportData()">Export Data</button>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // WebSocket connection
            let ws = null;
            let trafficChart = null;
            let trafficData = [];
            
            // Initialize chart
            function initChart() {
                const ctx = document.getElementById('traffic-chart').getContext('2d');
                trafficChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Packets/sec',
                            data: [],
                            borderColor: '#44ff44',
                            backgroundColor: 'rgba(68, 255, 68, 0.1)',
                            tension: 0.4
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true,
                                grid: { color: 'rgba(255, 255, 255, 0.2)' },
                                ticks: { color: 'white' }
                            },
                            x: {
                                grid: { color: 'rgba(255, 255, 255, 0.2)' },
                                ticks: { color: 'white' }
                            }
                        },
                        plugins: {
                            legend: { labels: { color: 'white' } }
                        }
                    }
                });
            }
            
            // Connect to WebSocket
            function connectWebSocket() {
                ws = new WebSocket('ws://localhost:8002/ws');
                
                ws.onopen = function() {
                    console.log('WebSocket connected');
                    document.getElementById('status').textContent = 'CONNECTED';
                    document.getElementById('status').className = 'status running';
                };
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    
                    if (data.type === 'stats') {
                        updateStats(data.data);
                        updateChart(data.data);
                    } else if (data.type === 'alert') {
                        addAlert(data.data);
                    }
                };
                
                ws.onclose = function() {
                    console.log('WebSocket disconnected');
                    document.getElementById('status').textContent = 'DISCONNECTED';
                    document.getElementById('status').className = 'status stopped';
                    
                    // Reconnect after 3 seconds
                    setTimeout(connectWebSocket, 3000);
                };
            }
            
            // Update statistics display
            function updateStats(stats) {
                document.getElementById('packet-rate').textContent = stats.packet_rate?.toFixed(1) || '0';
                document.getElementById('packet-count').textContent = stats.packet_count || '0';
                document.getElementById('unique-sources').textContent = stats.unique_sources || '0';
                document.getElementById('syn-ratio').textContent = (stats.syn_ratio * 100)?.toFixed(1) + '%' || '0%';
                document.getElementById('udp-ratio').textContent = (stats.udp_ratio * 100)?.toFixed(1) + '%' || '0%';
            }
            
            // Update traffic chart
            function updateChart(stats) {
                const now = new Date().toLocaleTimeString();
                trafficData.push(stats.packet_rate || 0);
                
                if (trafficData.length > 20) {
                    trafficData.shift();
                }
                
                trafficChart.data.labels = Array.from({length: trafficData.length}, (_, i) => '');
                trafficChart.data.datasets[0].data = trafficData;
                trafficChart.update('none');
            }
            
            // Add alert to display
            function addAlert(alert) {
                const container = document.getElementById('alerts-container');
                const alertDiv = document.createElement('div');
                alertDiv.className = `alert ${alert.severity.toLowerCase()}`;
                alertDiv.innerHTML = `
                    <strong>${alert.attack_type}</strong><br>
                    <small>${alert.description}</small><br>
                    <small>Confidence: ${(alert.confidence * 100).toFixed(1)}%</small>
                `;
                
                container.insertBefore(alertDiv, container.firstChild);
                
                // Keep only last 5 alerts
                while (container.children.length > 5) {
                    container.removeChild(container.lastChild);
                }
            }
            
            // Control functions
            function startDetection() {
                fetch('/api/stats').then(r => r.json()).then(console.log);
            }
            
            function stopDetection() {
                console.log('Stop detection requested');
            }
            
            function clearAlerts() {
                document.getElementById('alerts-container').innerHTML = '<p>No alerts detected</p>';
            }
            
            function exportData() {
                fetch('/api/alerts').then(r => r.json()).then(data => {
                    const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'ddos-alerts.json';
                    a.click();
                });
            }
            
            // Initialize
            initChart();
            connectWebSocket();
        </script>
    </body>
    </html>
    """

def main():
    """Main function to run the standalone dashboard."""
    print("Starting Standalone DDoS Detection Dashboard")
    print("=" * 50)
    print("Dashboard: http://localhost:8002")
    print("API: http://localhost:8002/api/stats")
    print("WebSocket: ws://localhost:8002/ws")
    print("Press Ctrl+C to stop")
    
    uvicorn.run(
        "standalone_dashboard:app",
        host="0.0.0.0",
        port=8002,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()

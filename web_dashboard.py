#!/usr/bin/env python3
"""
DDoS Detection Web Dashboard
===========================

Real-time web dashboard for monitoring DDoS attacks and network statistics.
Provides REST API and WebSocket updates for real-time monitoring.

Author: DDoS Detection System
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel

# Import our detector
from ddos_detector import DDoSDetector, NetworkStats, AttackAlert

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="DDoS Detection Dashboard", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detector instance
detector = DDoSDetector()
connected_clients = set()

# Pydantic models
class DetectionConfig(BaseModel):
    syn_flood_threshold: int = 1000
    udp_flood_threshold: int = 2000
    http_flood_threshold: int = 500
    syn_ratio_threshold: float = 0.1
    udp_ratio_threshold: float = 0.8
    http_ratio_threshold: float = 0.9

class MitigationAction(BaseModel):
    action: str  # "block_ip", "rate_limit", "redirect"
    target: str  # IP address or range
    duration: int = 300  # seconds

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

@app.post("/api/mitigate")
async def apply_mitigation(action: MitigationAction):
    """Apply mitigation action."""
    try:
        # Here you would implement actual mitigation logic
        # For now, just log the action
        logger.info(f"Mitigation action: {action.action} on {action.target} for {action.duration}s")
        
        # In a real system, you would:
        # - Block IPs using firewall rules
        # - Apply rate limiting
        # - Redirect traffic to honeypots
        # - Update router configurations
        
        return JSONResponse(content={
            "status": "success", 
            "message": f"Mitigation applied: {action.action} on {action.target}"
        })
    except Exception as e:
        logger.error(f"Error applying mitigation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Send periodic updates
            await asyncio.sleep(1)
            
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
        <title>DDoS Detection Dashboard</title>
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
                <h1>🛡️ DDoS Detection Dashboard</h1>
                <p>Real-time network monitoring and attack detection</p>
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
                ws = new WebSocket('ws://localhost:8000/ws');
                
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

async def start_detection_system():
    """Start the detection system in background."""
    detector.start_detection()
    logger.info("DDoS detection system started")

@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    await start_detection_system()

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    detector.packet_capture.stop_capture()
    logger.info("DDoS detection system stopped")

def main():
    """Main function to run the web dashboard."""
    print("🚀 Starting DDoS Detection Web Dashboard")
    print("=" * 50)
    print("📊 Dashboard: http://localhost:8000")
    print("📡 API: http://localhost:8000/api/stats")
    print("🔌 WebSocket: ws://localhost:8000/ws")
    print("🛑 Press Ctrl+C to stop")
    
    uvicorn.run(
        "web_dashboard:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()

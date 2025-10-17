# DDoS Detection and Mitigation System

A real-time DDoS attack detection and mitigation system using machine learning and deep learning techniques. It monitors network traffic, classifies suspicious activities, and automatically reroutes or blocks malicious traffic to ensure uninterrupted service availability.

## Features

- **Real-time Detection**: Monitors network traffic for SYN floods, UDP floods, HTTP floods, and other attack patterns
- **Machine Learning**: Uses multiple ML models (Random Forest, SVM, Neural Networks) for accurate attack classification
- **Web Dashboard**: Beautiful real-time dashboard with live statistics and alerts
- **REST API**: Complete API for integration with other systems
- **WebSocket Support**: Real-time updates for monitoring dashboards
- **Automatic Mitigation**: Configurable mitigation actions for detected attacks
- **Scalable Architecture**: Designed for high-performance network monitoring
- **Multiple Versions**: Standalone, Simple, and Full versions for different use cases

## Requirements

- Python 3.8+
- Windows 10/11, macOS, or Linux
- No admin privileges required for Standalone and Simple versions

## Quick Start

### Option 1: Standalone Version (Recommended - No Admin Required)
```bash
# Clone the repository
git clone https://github.com/AdityaBisht07/DDoS-Detection-and-Mitigation-System.git
cd DDoS-Detection-and-Mitigation-System

# Run the standalone system (all-in-one, no admin privileges)
python final_launcher.py
```

### Option 2: Simple Version (No Admin Required)
```bash
# Clone the repository
git clone https://github.com/AdityaBisht07/DDoS-Detection-and-Mitigation-System.git
cd DDoS-Detection-and-Mitigation-System

# Run the simple system
python run_fixed_system.py
```

### Option 3: Full Version (Requires Admin Privileges)
```bash
# Clone the repository
git clone https://github.com/AdityaBisht07/DDoS-Detection-and-Mitigation-System.git
cd DDoS-Detection-and-Mitigation-System

# Install dependencies
pip install -r requirements.txt

# Run the full system
python run_system.py
```

### Access Dashboard
- **Standalone Version**: http://localhost:8002 (Recommended)
- **Simple Version**: http://localhost:8001
- **Full Version**: http://localhost:8000

## Testing the System

To verify everything works correctly:

```bash
# Test the standalone version
python test_final_system.py

# Test individual components
python quick_test.py
```

## System Components

### Standalone Version (Recommended)
- `standalone_dashboard.py` - All-in-one solution with detector and dashboard
- `final_launcher.py` - Ultimate launcher with all improvements
- No separate processes needed - everything runs in one application
- Works without admin privileges
- Dashboard: http://localhost:8002

### Simple Version
- `simple_detector.py` - Working detector with simulated data
- `simple_dashboard.py` - Web dashboard (no admin required)
- `run_fixed_system.py` - Easy launcher for the working version
- Uses simulated network data
- Dashboard: http://localhost:8001

### Full Version (Advanced)
- `ddos_detector.py` - Core detection engine with real packet capture
- `web_dashboard.py` - Full web interface and API
- `ml_trainer.py` - ML model training
- `run_system.py` - System launcher
- Requires admin privileges for packet capture
- Dashboard: http://localhost:8000

### Core Detection Engine
- Real-time packet capture using Scapy
- Statistical analysis of network traffic
- Rule-based attack detection
- ML model integration for classification

### Web Dashboard
- FastAPI-based web interface
- Real-time statistics and charts
- Attack alerts and notifications
- Configuration management
- WebSocket for live updates

### ML Training
- Synthetic data generation for training
- Multiple ML model training
- Model evaluation and comparison
- Model persistence and loading

## Attack Detection

The system detects various types of DDoS attacks:

### SYN Flood
- **Detection**: High SYN packet rate with low SYN+ACK ratio
- **Threshold**: Configurable packet rate and ratio thresholds
- **Severity**: High

### UDP Flood
- **Detection**: High UDP packet rate with minimal TCP traffic
- **Threshold**: Configurable packet rate and UDP ratio
- **Severity**: High

### HTTP Flood
- **Detection**: High HTTP request rate from limited sources
- **Threshold**: Configurable request rate and source count
- **Severity**: Medium

## Troubleshooting

### Common Issues

1. **Unicode/Emoji Errors**: If you see encoding errors, the system will work but without emoji characters
2. **Port Already in Use**: Try a different port or stop other services
3. **Permission Denied**: Use the Standalone or Simple versions (no admin required)
4. **Dependencies Missing**: The system will automatically install them

### Getting Help

- Check the logs for detailed error messages
- Ensure Python 3.8+ is installed
- For Windows: Run PowerShell as Administrator if using Full version
- For testing: Use the Standalone version (recommended)

## 🔧 Configuration

### Detection Thresholds
```python
attack_thresholds = {
    'syn_flood': {'packet_rate': 1000, 'syn_ratio': 0.1},
    'udp_flood': {'packet_rate': 2000, 'udp_ratio': 0.8},
    'http_flood': {'packet_rate': 500, 'http_ratio': 0.9}
}
```

### ML Models
- Random Forest Classifier
- Gradient Boosting Classifier
- Support Vector Machine
- Neural Network (MLP)

## 📡 API Endpoints

### Statistics
- `GET /api/stats` - Current network statistics
- `GET /api/alerts` - Recent attack alerts

### Configuration
- `POST /api/config` - Update detection thresholds
- `POST /api/mitigate` - Apply mitigation actions

### WebSocket
- `ws://localhost:8000/ws` - Real-time updates

## 🚨 Mitigation Actions

The system supports various mitigation strategies:

- **IP Blocking**: Block malicious source IPs
- **Rate Limiting**: Apply rate limits to suspicious traffic
- **Traffic Redirection**: Redirect traffic to honeypots
- **Load Balancing**: Distribute traffic across multiple servers

## 📈 Monitoring

### Real-time Metrics
- Packets per second
- Unique source/destination counts
- Protocol ratios (SYN, UDP, HTTP)
- Connection statistics
- Attack confidence scores

### Alerts
- Real-time attack notifications
- Severity classification
- Source and target identification
- Confidence scoring

## 🔒 Security Considerations

- **Network Access**: Requires elevated privileges for packet capture
- **Data Privacy**: No sensitive data is stored or transmitted
- **Resource Usage**: Monitors system resources to prevent overload
- **False Positives**: Configurable thresholds to minimize false alarms

## 🛠️ Development

### Project Structure
```
DDoS-Detection-and-Mitigation-System/
├── ddos_detector.py      # Core detection engine
├── web_dashboard.py      # Web interface and API
├── ml_trainer.py         # ML model training
├── run_system.py         # System launcher
├── requirements.txt      # Python dependencies
├── models/               # Trained ML models
└── README.md            # This file
```

### Adding New Attack Types
1. Define attack pattern in `DDoSDetector`
2. Add detection logic in `_detect_*` methods
3. Update ML training data generator
4. Add UI components in dashboard

### Custom ML Models
1. Implement model in `ml_trainer.py`
2. Add to model configuration
3. Update prediction pipeline
4. Test with synthetic data

## 📚 Learning Resources

This system demonstrates:
- Network packet analysis
- Machine learning for security
- Real-time data processing
- Web API development
- System monitoring and alerting

## ⚠️ Disclaimer

This system is for educational and research purposes. Always ensure you have proper authorization before monitoring network traffic. The system should be used in controlled environments and not deployed in production without proper security review.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Happy DDoS Hunting! 🛡️**

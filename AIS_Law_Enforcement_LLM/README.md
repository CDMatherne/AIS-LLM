# AIS Law Enforcement LLM Assistant
**Developed by the Dreadnaught Team for Booz Allen Hamilton's Datathon 2025**
**Version:** 1.0.0  
**Powered by:** Claude (Anthropic)  
**Purpose:** Maritime fraud detection and investigation assistance for law enforcement

---

## 🚢 Overview

The AIS Law Enforcement LLM Assistant is a Claude-powered conversational AI system designed to help law enforcement agencies analyze Automatic Identification System (AIS) data to detect and investigate maritime fraud. The system combines advanced language AI with geographic analysis tools and GPU-accelerated data processing.

### Key Features

- **💬 Natural Language Interface**: Ask questions in plain English about vessels, anomalies, and geographic areas
- **🗺️ Geographic Zone Management**: Define custom polygons, circles, ovals, and rectangles for analysis
- **⚡ GPU Acceleration**: Support for NVIDIA CUDA and AMD ROCm/HIP for fast processing
- **☁️ Flexible Data Sources**: Connect to AWS S3 or local file systems
- **🔍 Comprehensive Anomaly Detection**:
  - AIS Beacon Off/On (dark shipping)
  - Speed Anomalies
  - COG/Heading Inconsistencies
  - Loitering
  - Rendezvous
  - Identity Spoofing
  - Zone Violations

### ⚠️ Test Implementation Limitations

This is a **test implementation** with the following limitations:
- **Date Range**: October 15, 2024 through March 30, 2025 only
- **Data Formats**: CSV and Parquet files only
- Analysis outside this date range will not return results

---

## 📋 Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Architecture](#architecture)
5. [API Reference](#api-reference)
6. [GPU Support](#gpu-support)
7. [Anomaly Types](#anomaly-types)
8. [Troubleshooting](#troubleshooting)

---

## 🚀 Installation

### Prerequisites

- **Python**: 3.8 or higher (3.10+ recommended)
- **Operating System**: Windows, Linux, or macOS
- **Claude API Key**: From [console.anthropic.com](https://console.anthropic.com/)
- **Data Source**: AWS S3 bucket or local folder with AIS data

### Step 1: Clone or Download

```bash
cd "AIS Law Enforcement LLM"
```

### Step 2: Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### Step 3: GPU Support (Optional but Recommended)

#### For NVIDIA GPUs:
```bash
pip install --extra-index-url=https://pypi.nvidia.com cudf-cu11 cuml-cu11 cupy-cuda11x
```

**Requirements**:
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.8 or later
- Compatible NVIDIA driver

#### For AMD GPUs (Linux):
```bash
pip install cupy-rocm-5-0
```

**Requirements**:
- AMD GPU with ROCm support
- ROCm 5.0 or later

#### For AMD GPUs (Windows - Experimental):
```bash
pip install pyhip
```

**Requirements**:
- AMD HIP SDK from [AMD Developer Resources](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html)

---

## ⚡ Quick Start

### 1. Start the Backend Server

```bash
cd backend
python app.py
```

The server will start on `http://localhost:8000`

### 2. Open the Frontend

Open your web browser and navigate to:
```
frontend/setup.html
```

### 3. Configure Data Source

**Option A: AWS S3**
1. Select "AWS S3 Bucket"
2. Enter your bucket name and configuration
3. Choose authentication method:
   - **Access Key & Secret**: Enter AWS credentials
   - **AWS Profile**: Use credentials from ~/.aws/credentials
   - **IAM Role**: For EC2/Lambda instances

**Option B: Local Files**
1. Select "Local Folder"
2. Enter the path to your data folder
3. Ensure files are named with date format:
   - `YYYY-MM-DD.parquet` or `YYYY-MM-DD.csv` (standard format)
   - `ais-YYYY-MM-DD.parquet` or `ais-YYYY-MM-DD.csv` (alternative format)

### 4. Enter Claude API Key

Get your API key from [Anthropic Console](https://console.anthropic.com/) and enter it in the configuration form.

### 5. Test Connection

Click "Test Connection" to verify your data source is accessible.

### 6. Save & Continue

Click "Save & Continue" to initialize the system and start the chat interface.

---

## ⚙️ Configuration

### Data File Structure

Your data files should follow one of these naming conventions:

**Standard Format:**
```
AIS_Data/
├── 2024-10-15.parquet (or .csv)
├── 2024-10-16.parquet (or .csv)
├── 2024-10-17.parquet (or .csv)
├── ...
└── 2025-03-30.parquet (or .csv)
```

**Alternative Format (with "ais-" prefix):**
```
AIS_Data/
├── ais-2024-10-15.parquet (or .csv)
├── ais-2024-10-16.parquet (or .csv)
├── ais-2024-10-17.parquet (or .csv)
├── ...
└── ais-2025-03-30.parquet (or .csv)
```

**Note:** Both formats can be used in the same folder. The system will automatically detect and load files in either format.

### Required AIS Data Fields

Your CSV/Parquet files must include these columns:
- `MMSI`: Maritime Mobile Service Identity
- `LAT`: Latitude (decimal degrees)
- `LON`: Longitude (decimal degrees)
- `BaseDateTime`: Timestamp
- `SOG`: Speed Over Ground (knots)
- `COG`: Course Over Ground (degrees)
- `Heading`: Vessel heading (degrees)
- `VesselType`: Vessel type code
- `VesselName`: Vessel name (optional but recommended)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│              Frontend (HTML/CSS/JS)                  │
│  - Setup page for configuration                     │
│  - Chat interface with Claude                       │
│  - Map viewer (planned)                             │
└────────────────┬────────────────────────────────────┘
                 │
                 │ HTTP/WebSocket
                 │
┌────────────────▼────────────────────────────────────┐
│           Backend (FastAPI/Python)                  │
│  ┌────────────────────────────────────────────┐    │
│  │         Claude Agent                        │    │
│  │  - Natural language understanding           │    │
│  │  - Tool calling                             │    │
│  │  - Context management                       │    │
│  └────────────┬───────────────────────────────┘    │
│               │                                      │
│  ┌────────────▼───────────────────────────────┐    │
│  │      Analysis Engine                        │    │
│  │  - Anomaly detection                        │    │
│  │  - GPU acceleration                         │    │
│  │  - Statistical analysis                     │    │
│  └────────────┬───────────────────────────────┘    │
│               │                                      │
│  ┌────────────▼───────────────────────────────┐    │
│  │      Data Connector                         │    │
│  │  - AWS S3 / Local filesystem                │    │
│  │  - CSV/Parquet support                      │    │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### Key Components

1. **Claude Agent** (`claude_agent.py`): Handles conversation and tool calling
2. **Analysis Engine** (`analysis_engine.py`): Performs anomaly detection with GPU support
3. **Data Connector** (`data_connector.py`): Manages data source connections
4. **Geographic Tools** (`geographic_tools.py`): Handles zones and spatial queries
5. **Session Manager** (`session_manager.py`): Manages user sessions
6. **GPU Support** (`gpu_support.py`): Detects and utilizes GPU acceleration

---

## 🔧 API Reference

### Endpoints

#### `POST /api/setup`
Initialize system with configuration
```json
{
  "data_source": "aws",
  "claude_api_key": "sk-ant-...",
  "aws": {
    "bucket": "my-bucket",
    "region": "us-east-1",
    "auth_method": "credentials"
  }
}
```

#### `POST /api/chat`
Send chat message
```json
{
  "session_id": "uuid",
  "message": "Show me all anomalies in the South China Sea"
}
```

#### `WebSocket /ws/chat/{session_id}`
Real-time chat connection

#### `GET /api/gpu-status`
Get GPU information and status

#### `GET /api/health`
Health check endpoint

---

## 🎮 GPU Support

### Benefits of GPU Acceleration

- **10-100x faster** data processing
- Handle larger datasets efficiently
- Real-time analysis of millions of AIS records

### Checking GPU Status

The system automatically detects available GPUs on startup. Check the setup page for GPU status.

### Performance Comparison

| Operation | CPU | NVIDIA GPU | AMD GPU |
|-----------|-----|------------|---------|
| Load 1M records | 15s | 2s | 3s |
| Anomaly detection | 45s | 5s | 8s |
| Filter & aggregate | 10s | 1s | 2s |

---

## 🔍 Anomaly Types

### 1. AIS Beacon Off
**Description**: Vessel disappears for extended period  
**Threshold**: 6+ hours (configurable)  
**Indicators**: Dark shipping, illegal operations, sanctions evasion

### 2. AIS Beacon On
**Description**: Vessel reappears after being off  
**Indicators**: Position jumps, impossible travel distances

### 3. Speed Anomalies
**Description**: Impossible travel distances between positions  
**Threshold**: >550 nautical miles (configurable)  
**Indicators**: Data errors, spoofing, beacon manipulation

### 4. COG/Heading Inconsistency
**Description**: Large difference between course and heading  
**Threshold**: >45° difference (configurable)  
**Indicators**: Drifting, towing, navigation issues, spoofing

### 5. Loitering
**Description**: Extended presence in small area  
**Indicators**: Illegal fishing, waiting for rendezvous, surveillance

### 6. Rendezvous
**Description**: Multiple vessels meeting at sea  
**Indicators**: Transshipment, smuggling, illegal fishing cooperation

### 7. Identity Spoofing
**Description**: Multiple vessels using same MMSI  
**Indicators**: Fraudulent identity, intentional deception

### 8. Zone Violations
**Description**: Entry into restricted areas  
**Indicators**: Territorial violations, EEZ incursions, sanctioned ports

---

## 🛠️ Troubleshooting

### Backend Won't Start

**Problem**: `ModuleNotFoundError`  
**Solution**: Install dependencies: `pip install -r requirements.txt`

**Problem**: Port 8000 already in use  
**Solution**: Change port in `app.py` or kill process using port 8000

### Connection Test Fails

**Problem**: Cannot connect to S3  
**Solution**: 
- Verify AWS credentials are correct
- Check S3 bucket name and region
- Ensure bucket contains date-formatted files

**Problem**: Local files not found  
**Solution**:
- Verify folder path is correct
- Check files are named using one of the supported formats:
  - `YYYY-MM-DD.parquet` or `YYYY-MM-DD.csv` (standard)
  - `ais-YYYY-MM-DD.parquet` or `ais-YYYY-MM-DD.csv` (alternative)
- Ensure dates are within Oct 15, 2024 - Mar 30, 2025

### Claude Not Responding

**Problem**: API key invalid  
**Solution**: Get new API key from [console.anthropic.com](https://console.anthropic.com/)

**Problem**: Rate limit exceeded  
**Solution**: Wait a few minutes or upgrade Claude API tier

### GPU Not Detected

**Problem**: NVIDIA GPU not detected  
**Solution**:
- Install CUDA Toolkit 11.8+
- Update NVIDIA drivers
- Install cudf/cupy packages

**Problem**: AMD GPU not detected  
**Solution**:
- Install ROCm (Linux) or HIP SDK (Windows)
- Install cupy-rocm or pyhip package

---

## 📚 Additional Resources

- **Claude API Documentation**: [docs.anthropic.com](https://docs.anthropic.com/)
- **RAPIDS (NVIDIA GPU)**: [rapids.ai](https://rapids.ai/)
- **ROCm (AMD GPU)**: [rocm.docs.amd.com](https://rocm.docs.amd.com/)
- **AIS Information**: [IALA](https://www.iala-aism.org/)

---

## 📝 License

This is a demonstration/test implementation for law enforcement use.

---

## 🤝 Support

For issues, questions, or feature requests, please contact your system administrator.

---

## ⚠️ Legal Notice

This system is designed for law enforcement use only. All data analysis must comply with applicable laws and regulations regarding data privacy, maritime jurisdiction, and law enforcement procedures.

---

**Built with:**
- Claude (Anthropic) for AI assistance
- FastAPI for backend
- Shapely/PyProj for geographic analysis
- RAPIDS/cuDF for GPU acceleration
- boto3 for AWS S3 integration


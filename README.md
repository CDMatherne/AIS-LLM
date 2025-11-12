# AIS Law Enforcement LLM Assistant

A sophisticated AI-powered system designed to assist law enforcement in detecting and investigating maritime fraud using Automatic Identification System (AIS) data. The system combines advanced language AI (Claude) with geographic analysis tools, GPU-accelerated data processing, and comprehensive visualization capabilities.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [File Structure](#file-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Data Cache System](#data-cache-system)
- [GPU Support](#gpu-support)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🎯 Overview

This system provides law enforcement agencies with an intelligent assistant for analyzing AIS (Automatic Identification System) data to detect suspicious maritime activities. It uses Claude AI to understand natural language queries and automatically executes complex analyses, generates visualizations, and creates investigation reports.

### Key Capabilities

- **Natural Language Interaction**: Ask questions in plain English about vessels, anomalies, and geographic areas
- **Anomaly Detection**: Identifies 8 types of maritime anomalies including dark shipping, spoofing, rendezvous, and more
- **Geographic Analysis**: Define and monitor custom geographic zones (polygons, circles, rectangles, ovals)
- **Visualization**: Generate interactive maps, charts, and custom visualizations
- **Export Capabilities**: Export professional reports in CSV, Excel, and HTML formats
- **Vessel Tracking**: Track specific vessels by MMSI across time periods
- **High-Risk Identification**: Automatically identify vessels with the most anomalies for investigation prioritization

### Test Implementation Notice

⚠️ **This is a test implementation with limited data availability:**
- Data range: **October 15, 2024 through March 30, 2025**
- Queries outside this date range will not return results
- Designed for demonstration and testing purposes

## ✨ Key Features

### 1. AI-Powered Analysis
- Claude AI integration for natural language understanding
- Automatic tool selection and execution
- Context-aware responses based on maritime domain knowledge
- Interactive workflow with popup dialogs for user input

### 2. Anomaly Detection
Detects 8 types of maritime anomalies:
- **AIS Beacon Off**: Vessel disappears for extended periods (>6 hours)
- **AIS Beacon On**: Vessel reappears after being off (position jumps)
- **Speed Anomalies**: Impossible travel distances between positions
- **COG/Heading Inconsistency**: Large difference between Course Over Ground and Heading
- **Loitering**: Extended presence in small area
- **Rendezvous**: Multiple vessels meeting at sea
- **Identity Spoofing**: Multiple vessels using same MMSI
- **Zone Violations**: Entry into restricted or monitored areas

### 3. Geographic Tools
- 24 predefined water bodies (oceans, seas, gulfs, straits, canals)
- Custom zone creation (polygon, circle, rectangle, oval)
- Geographic filtering and analysis
- Location identification for vessels and coordinates

### 4. Visualization & Export
- Interactive Folium maps with clustering and heatmaps
- Multiple chart types (bar, pie, 3D, scatterplot, time series)
- Custom visualization creation with Python code
- Export to CSV, Excel, and HTML formats
- Timestamped outputs to prevent overwrites

### 5. Performance Optimization
- **Data Cache System**: Pre-loads all data for 60-70% faster analysis
- **GPU Acceleration**: Optional NVIDIA/AMD GPU support for faster processing
- **Background Sync**: Automatic data synchronization every 6 hours
- **Optimized Data Formats**: Parquet format with compression

## 🏗️ Architecture

The system follows a client-server architecture:

```
┌─────────────────┐
│   Frontend      │  HTML/JavaScript Web Interface
│   (Browser)     │  - Chat interface
│                 │  - Setup wizard
│                 │  - File viewer
└────────┬────────┘
         │ HTTP/WebSocket
         │
┌────────▼────────┐
│   Backend       │  FastAPI Server
│   (Python)      │  - REST API endpoints
│                 │  - WebSocket for real-time chat
│                 │  - Tool execution engine
└────────┬────────┘
         │
    ┌────┴────┬──────────┬──────────────┐
    │         │          │              │
┌───▼───┐ ┌──▼───┐ ┌───▼────┐ ┌───────▼────┐
│ Claude│ │ Data │ │ Analysis│ │Visualization│
│  AI   │ │Cache │ │ Engine  │ │  Engine    │
└───────┘ └──────┘ └─────────┘ └────────────┘
```

### Component Overview

1. **Frontend**: Single-page web application with chat interface
2. **Backend API**: FastAPI server handling requests and WebSocket connections
3. **Claude Agent**: Conversational AI that understands maritime domain
4. **Analysis Engine**: Core anomaly detection logic with GPU support
5. **Data Connector**: Universal connector for AWS S3 and local filesystems
6. **Data Cache**: High-performance caching layer for fast data access
7. **Visualization Engine**: Dynamic chart and map generation
8. **Tool Handlers**: Registry-based system for executing Claude's tool calls

## 📁 File Structure

### Backend Files

#### Core Application
- **`backend/app.py`** (2,552 lines)
  - Main FastAPI application
  - API endpoints for setup, chat, analysis, exports, visualizations
  - WebSocket endpoint for real-time chat
  - Session management integration
  - Data cache initialization and background sync
  - File management endpoints
  - GPU status endpoints

- **`backend/claude_agent.py`** (1,340 lines)
  - Claude AI integration and conversation management
  - Model auto-detection (finds best available Claude model)
  - System prompt with maritime domain knowledge
  - Tool definitions for Claude function calling
  - Conversation history validation and cleanup
  - Tool result processing

- **`backend/analysis_engine.py`** (891 lines)
  - Core anomaly detection logic
  - GPU acceleration support (NVIDIA/AMD)
  - Progress reporting for long operations
  - Vessel tracking and history retrieval
  - High-risk vessel identification
  - Statistics generation

- **`backend/data_connector.py`** (396 lines)
  - Universal data connector for AWS S3 and local filesystems
  - Supports CSV and Parquet formats
  - Date range validation (enforces test implementation limits)
  - File pattern matching (supports `ais-YYYY-MM-DD` and `YYYY-MM-DD` formats)
  - Cache integration for fast data access

- **`backend/data_cache.py`**
  - High-performance data caching system
  - Pre-loads all available data on startup
  - Background synchronization (every 6 hours)
  - Indexed lookups by date, vessel type, and MMSI
  - Optimized Parquet storage with compression

#### Geographic & Analysis Tools
- **`backend/geographic_tools.py`**
  - Geographic zone management (polygon, circle, rectangle, oval)
  - 24 predefined water bodies (oceans, seas, gulfs, straits, canals)
  - Location identification and filtering
  - Water body statistics and lookups

- **`backend/vessel_types.py`**
  - 58 AIS vessel type definitions (codes 20-99)
  - 8 categories: WIG, Special, HSC, Special Purpose, Passenger, Cargo, Tanker, Other
  - Hazardous cargo classification
  - Vessel type filtering utilities

- **`backend/water_bodies.py`**
  - Definitions of 24 major water bodies worldwide
  - Geographic boundaries and metadata
  - Parent-child relationships (e.g., Mediterranean Sea → Atlantic Ocean)

- **`backend/restricted_zones.py`**
  - Restricted area definitions
  - Zone violation detection

#### Visualization & Export
- **`backend/visualization_engine.py`**
  - Dynamic visualization creation system
  - Code execution for custom visualizations
  - Visualization registry and templates
  - Supports matplotlib, plotly, and folium

- **`backend/map_creator.py`**
  - Interactive Folium map generation
  - Anomaly mapping with clustering
  - Vessel track visualization
  - Heatmap generation

- **`backend/chart_creator.py`**
  - Statistical chart generation
  - Multiple chart types (bar, pie, 3D, scatterplot, time series)
  - Anomaly distribution visualization
  - Top vessels charts

- **`backend/export_utils.py`**
  - CSV and Excel export functionality
  - Formatted reports with statistics
  - Timestamped file naming

#### Tool Handlers (`backend/tool_handlers/`)
Registry-based system for executing Claude's tool calls:

- **`registry.py`**: Tool registration and handler lookup
- **`base.py`**: Base handler class and utilities
- **`dependencies.py`**: Dependency injection for session manager
- **`analysis_handlers.py`**: Anomaly analysis tool handlers
- **`geographic_handlers.py`**: Geographic zone and location handlers
- **`chart_handlers.py`**: Chart generation handlers
- **`map_handlers.py`**: Map generation handlers
- **`export_handlers.py`**: Export functionality handlers
- **`visualization_handlers.py`**: Custom visualization handlers
- **`ui_handlers.py`**: UI interaction handlers (popups, selections)

#### System Management
- **`backend/session_manager.py`**
  - User session management
  - Session timeout handling (default 2 hours)
  - Analysis result storage
  - Zone management per session

- **`backend/temp_file_manager.py`**
  - Temporary file creation and cleanup
  - Session-based file management
  - Automatic cleanup on session end

- **`backend/interaction_logger.py`**
  - Interaction logging for debugging
  - Tool execution tracking
  - Performance metrics

- **`backend/gpu_support.py`**
  - GPU detection (NVIDIA CUDA, AMD ROCm)
  - GPU library initialization
  - Processing backend selection
  - Installation instructions

- **`backend/diagnostic_tool.py`**
  - System diagnostics and health checks
  - Configuration validation

### Frontend Files

- **`frontend/index.html`** (1,714 lines)
  - Main chat interface
  - WebSocket connection management
  - Message handling and display
  - Popup dialog system (date picker, vessel type selector, etc.)
  - Debug panel (Ctrl+D)
  - Environment variable configuration support

- **`frontend/setup.html`**
  - Configuration wizard
  - AWS S3 and local data source setup
  - Connection testing
  - Claude API key configuration

- **`frontend/files.html`**
  - File browser for generated outputs
  - View and download maps, charts, exports
  - File categorization (maps, charts, exports)

- **`frontend/welcome_preview.html`**
  - Welcome screen preview

- **`frontend/css/styles.css`**
  - Styling for all frontend pages
  - Responsive design
  - Modal and popup styles

- **`frontend/js/setup.js`**
  - Setup page JavaScript logic
  - Form validation
  - Connection testing
  - Configuration storage

### Configuration & Documentation

- **`backend/requirements.txt`**: Python dependencies with GPU support options
- **`DATA_CACHE_IMPLEMENTATION.md`**: Data cache system documentation
- **`SEED_VISUALIZATIONS.py`**: Script to seed visualization registry
- **`SFDImage.png`**: Application logo

## 📦 Requirements

### Python Dependencies

All dependencies are listed in `backend/requirements.txt`:

#### Core Web Framework
- `fastapi==0.104.1` - Modern web framework for building APIs
- `uvicorn[standard]==0.24.0` - ASGI server for FastAPI
- `python-multipart==0.0.6` - File upload support
- `websockets==12.0` - WebSocket support for real-time chat

#### Claude AI
- `anthropic==0.7.7` - Anthropic Claude API client

#### Data Processing
- `pandas>=2.2.3` - Data manipulation and analysis (Python 3.14+ compatible)
- `pyarrow==14.0.1` - Parquet file support
- `numpy==1.26.2` - Numerical computing

#### AWS Support
- `boto3==1.29.7` - AWS SDK for Python
- `botocore==1.32.7` - Low-level AWS service access

#### Geographic/Spatial
- `shapely==2.0.2` - Geometric operations
- `pyproj==3.6.1` - Cartographic projections

#### Utilities
- `python-dateutil==2.8.2` - Date parsing and manipulation
- `pydantic==2.5.0` - Data validation
- `requests==2.31.0` - HTTP library
- `python-dotenv==1.0.0` - Environment variable management

#### Visualization & Export
- `matplotlib==3.8.2` - Static plotting
- `seaborn==0.13.0` - Statistical visualization
- `plotly==5.18.0` - Interactive plotting
- `folium==0.15.1` - Interactive maps
- `openpyxl==3.1.2` - Excel file support
- `xlsxwriter==3.1.9` - Excel writing
- `Pillow==10.1.0` - Image processing
- `branca==0.7.0` - Folium dependency

#### Optional GPU Support
GPU acceleration is optional but recommended for performance:

**NVIDIA GPU (CUDA):**
- `cudf-cu11==23.10.0` - GPU-accelerated DataFrames
- `cuml-cu11==23.10.0` - GPU-accelerated ML
- `cupy-cuda11x==12.3.0` - NumPy-like GPU arrays
- `dask-cudf==23.10.0` - Distributed GPU DataFrames

**AMD GPU (ROCm/HIP):**
- `cupy-rocm-5-0==12.3.0` - AMD GPU support (Linux primarily)
- `pyhip==1.0.0` - Direct HIP runtime access (Windows experimental)

### System Requirements

- **Python**: 3.10+ (3.14+ recommended for pandas compatibility)
- **Operating System**: Windows, macOS, or Linux
- **Memory**: 8GB RAM minimum (16GB+ recommended for large datasets)
- **Storage**: Sufficient space for data cache (typically 20-30% of data size)
- **Network**: Internet connection for Claude API and AWS S3 access

### Optional Requirements

- **NVIDIA GPU**: CUDA Toolkit 11.8+ and compatible driver for GPU acceleration
- **AMD GPU**: ROCm 5.0+ (Linux) or AMD HIP SDK (Windows experimental)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd AIS_Law_Enforcement_LLM
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

#### Basic Installation (CPU only)
```bash
cd backend
pip install -r requirements.txt
```

#### NVIDIA GPU Installation
1. Ensure CUDA Toolkit 11.8+ and compatible NVIDIA driver are installed
2. Uncomment NVIDIA GPU lines in `requirements.txt`
3. Install:
```bash
pip install -r requirements.txt
pip install --extra-index-url=https://pypi.nvidia.com cudf-cu11 cuml-cu11
```

#### AMD GPU Installation (Linux)
1. Ensure ROCm 5.0+ is installed
2. Uncomment AMD cupy-rocm line in `requirements.txt`
3. Install:
```bash
pip install -r requirements.txt
```

#### AMD GPU Installation (Windows - Experimental)
1. Install AMD HIP SDK from: https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html
2. Uncomment PyHIP line in `requirements.txt`
3. Install:
```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import fastapi, anthropic, pandas; print('Installation successful!')"
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root (optional but recommended):

```env
# Claude API Configuration
CLAUDE_API_KEY=sk-ant-api03-...

# Data Source Configuration
DATA_SOURCE=aws  # or 'local'

# AWS S3 Configuration (if DATA_SOURCE=aws)
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_SESSION_TOKEN=...  # Optional, for temporary credentials
AWS_REGION=us-east-1
AWS_BUCKET=my-ais-data-bucket
AWS_PREFIX=ais-data/parquet/  # Optional prefix

# Local Data Configuration (if DATA_SOURCE=local)
LOCAL_DATA_PATH=/path/to/ais/data
LOCAL_FILE_FORMAT=auto  # or 'parquet' or 'csv'

# Output Directory (optional, defaults to Downloads/AISDS_Output)
OUTPUT_DIRECTORY=/path/to/output
```

### Manual Configuration

If not using environment variables, configure through the web interface:

1. Start the backend server
2. Open `frontend/setup.html` in a web browser
3. Fill in:
   - Claude API key
   - Data source (AWS S3 or local)
   - Data source credentials/path
4. Test connection
5. Save configuration

### Claude API Key

Get your API key from: https://console.anthropic.com/

- Free tier available
- Model auto-detection selects best available model
- Opus models excluded for cost efficiency (uses Sonnet/Haiku)

## 🎮 Usage

### Starting the Server

```bash
cd backend
python app.py
```

Or using uvicorn directly:

```bash
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The server will start on `http://localhost:8000`

### Accessing the Web Interface

1. Open `frontend/index.html` in a web browser
2. If not configured, you'll be redirected to `setup.html`
3. Complete setup wizard
4. Start chatting with the AI assistant

### Example Queries

**Analysis Requests:**
- "Analyze anomalies from October 15 to October 17, 2024"
- "Find vessels with beacon off anomalies in the Mediterranean Sea"
- "Show me top 10 high-risk vessels from last week"

**Vessel Tracking:**
- "Track vessel MMSI 123456789 from October 15 to October 20"
- "What is the location of vessel 123456789?"

**Geographic Queries:**
- "Create a monitoring zone around coordinates -74.0, 40.7 with 10 nautical mile radius"
- "Which water body is vessel 123456789 in?"
- "List all available water bodies"

**Visualization Requests:**
- "Create a map showing all anomalies"
- "Generate a heatmap of anomaly density"
- "Create a chart showing anomaly types distribution"
- "Export results to Excel"

### Interactive Workflow

The system uses an interactive workflow for analysis:

1. **Date Selection**: Popup dialog to select date range
2. **Vessel Type Selection**: Checkboxes to select vessel types
3. **Anomaly Type Selection**: Checkboxes to select anomaly types
4. **Output Selection**: Checkboxes to select output formats
5. **Analysis Execution**: Automatic analysis with progress updates
6. **Output Generation**: Timestamped files saved to output folder

### Output Files

All outputs are saved to `Downloads/AISDS_Output/` (or custom output directory):

- **Maps**: `Path_Maps/` folder (HTML files)
- **Charts**: `Charts/` folder (PNG, HTML files)
- **Exports**: Root folder (CSV, Excel files)
- **Files are timestamped**: `filename_YYYYMMDD_HHMMSS.ext`

View files: Open `frontend/files.html` in browser

## 📡 API Documentation

### REST Endpoints

#### Setup & Configuration
- `POST /api/setup` - Initialize system with configuration
- `POST /api/setup-from-env` - Initialize from environment variables
- `GET /api/check-env-config` - Check environment configuration
- `POST /api/test-connection` - Test data source connection
- `POST /api/test-aws` - Test AWS S3 connection
- `POST /api/test-local-path` - Test local directory access

#### Chat & Analysis
- `POST /api/chat` - Send chat message (HTTP)
- `WebSocket /ws/chat/{session_id}` - Real-time chat (WebSocket)
- `GET /api/sessions/{session_id}/info` - Get session information

#### Data Cache
- `GET /api/cache/status` - Get cache status and statistics
- `POST /api/cache/sync?force=false` - Manually trigger cache sync

#### Exports
- `POST /api/export/csv` - Export to CSV
- `POST /api/export/excel` - Export to Excel
- `GET /api/exports/list` - List available exports
- `GET /api/exports/download/{filename}` - Download export file

#### Visualizations
- `POST /api/visualizations/create` - Create custom visualization
- `POST /api/visualizations/execute/{viz_id}` - Execute saved visualization
- `GET /api/visualizations/list` - List custom visualizations
- `GET /api/visualizations/templates` - Get visualization templates
- `POST /api/visualizations/{viz_id}/rate` - Rate visualization
- `DELETE /api/visualizations/{viz_id}` - Delete visualization

#### Maps
- `POST /api/maps/all-anomalies` - Create all anomalies map
- `POST /api/maps/vessel-track` - Create vessel track map
- `POST /api/maps/heatmap` - Create anomaly heatmap

#### Charts
- `POST /api/charts/anomaly-types` - Create anomaly types chart
- `POST /api/charts/top-vessels` - Create top vessels chart
- `POST /api/charts/by-date` - Create anomalies by date chart
- `POST /api/charts/3d-bar` - Create 3D bar chart
- `POST /api/charts/scatterplot` - Create scatterplot

#### Files
- `GET /api/files/list` - List generated files
- `GET /api/files/view/{file_id}` - View file
- `GET /api/files/download/{file_id}` - Download file

#### System
- `GET /` - Root endpoint with service info
- `GET /api/health` - Health check
- `GET /api/gpu-status` - GPU status and information

### WebSocket Protocol

Connect to: `ws://localhost:8000/ws/chat/{session_id}`

**Client → Server:**
```json
{
  "message": "Analyze anomalies from October 15 to 17"
}
```

**Server → Client:**
```json
{
  "type": "message",
  "content": "I'll analyze anomalies for that date range..."
}
```

```json
{
  "type": "tools_executing",
  "tools": ["run_anomaly_analysis"]
}
```

```json
{
  "type": "tool_result",
  "tool_name": "run_anomaly_analysis",
  "result": {...}
}
```

```json
{
  "type": "progress",
  "stage": "loading_data",
  "message": "Loading data from 3 files...",
  "data": {...}
}
```

## 💾 Data Cache System

The data cache system dramatically improves performance by pre-loading all available data:

### Features

- **Automatic Pre-loading**: Loads all data on startup (one-time 5-15 minutes)
- **Background Sync**: Checks for new data every 6 hours
- **Fast Access**: 60-70% faster than direct S3 loading
- **Indexed Lookups**: Fast queries by date, vessel type, and MMSI
- **Transparent**: Automatically used by data connector

### Cache Status

Check cache status:
```bash
curl http://localhost:8000/api/cache/status
```

Manually sync cache:
```bash
curl -X POST http://localhost:8000/api/cache/sync
```

Force full reload:
```bash
curl -X POST "http://localhost:8000/api/cache/sync?force=true"
```

### Cache Storage

- **Location**: `backend/data_cache/`
- **Files**:
  - `ais_data_cache.parquet` - Optimized data storage
  - `ais_index_cache.pkl` - Fast lookup indexes
  - `cache_metadata.json` - Cache metadata

See `DATA_CACHE_IMPLEMENTATION.md` for detailed documentation.

## 🎮 GPU Support

### Checking GPU Status

```bash
curl http://localhost:8000/api/gpu-status
```

### NVIDIA GPU Setup

1. Install CUDA Toolkit 11.8+ from NVIDIA
2. Install compatible NVIDIA driver
3. Uncomment NVIDIA lines in `requirements.txt`
4. Install GPU libraries:
```bash
pip install --extra-index-url=https://pypi.nvidia.com cudf-cu11 cuml-cu11 cupy-cuda11x dask-cudf
```

### AMD GPU Setup (Linux)

1. Install ROCm 5.0+ from AMD
2. Uncomment AMD cupy-rocm line in `requirements.txt`
3. Install:
```bash
pip install cupy-rocm-5-0
```

### Performance Benefits

- **Haversine calculations**: 5-10x faster on GPU
- **Large dataset processing**: 2-3x faster overall
- **Memory efficiency**: Better handling of large DataFrames

## 🐛 Troubleshooting

### Backend Won't Start

**Issue**: Import errors or missing dependencies
- **Solution**: Ensure virtual environment is activated and all dependencies are installed
```bash
pip install -r backend/requirements.txt
```

**Issue**: Port 8000 already in use
- **Solution**: Change port in `app.py` or kill process using port 8000

### Frontend Can't Connect

**Issue**: "Connection error" in browser
- **Solution**: Ensure backend server is running on `http://localhost:8000`
- Check browser console for errors (F12)

**Issue**: CORS errors
- **Solution**: Backend CORS is configured to allow all origins. If issues persist, check firewall settings.

### Data Loading Issues

**Issue**: "No data found for specified date range"
- **Solution**: 
  - Verify date range is within October 15, 2024 - March 30, 2025
  - Check data files exist in S3/local path
  - Verify file naming: `ais-YYYY-MM-DD.parquet` or `YYYY-MM-DD.parquet`

**Issue**: AWS S3 connection fails
- **Solution**:
  - Verify AWS credentials are correct
  - Check bucket name and prefix
  - Ensure IAM permissions allow S3 read access
  - Test connection via `/api/test-aws` endpoint

**Issue**: Local file access fails
- **Solution**:
  - Verify path exists and is accessible
  - Check file permissions
  - Ensure files match naming pattern

### Claude API Issues

**Issue**: "No Claude models available"
- **Solution**:
  - Verify API key is correct
  - Check account status at https://console.anthropic.com/
  - Ensure API key has model access
  - Check for rate limits or quota exceeded

**Issue**: Slow responses
- **Solution**:
  - System uses fastest available model (Sonnet/Haiku, not Opus)
  - Conversation history is limited to 50 messages
  - Consider using data cache for faster data loading

### Cache Issues

**Issue**: Cache not initializing
- **Solution**:
  - Check AWS credentials are valid
  - Verify S3 bucket access
  - Check logs for error messages
  - Cache only works with AWS S3 data source

**Issue**: Cache out of date
- **Solution**:
  - Manually trigger sync: `POST /api/cache/sync`
  - Wait for background sync (every 6 hours)
  - Force full reload: `POST /api/cache/sync?force=true`

### GPU Issues

**Issue**: GPU not detected
- **Solution**:
  - Verify GPU drivers are installed
  - Check GPU libraries are installed correctly
  - System works fine on CPU if GPU unavailable
  - Check `/api/gpu-status` for details

**Issue**: GPU libraries fail to import
- **Solution**:
  - Verify CUDA/ROCm installation
  - Check Python version compatibility
  - System falls back to CPU automatically

### Session Issues

**Issue**: Session expired
- **Solution**: Sessions timeout after 2 hours of inactivity. Create a new session via setup page.

**Issue**: Lost conversation history
- **Solution**: Conversation history is stored in session. If session expires, history is lost. Consider exporting important results.

## 📝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

### Code Style

- Follow PEP 8 for Python code
- Use type hints where possible
- Add docstrings to functions and classes
- Comment complex logic

### Testing

- Test with sample AIS data
- Verify all endpoints work
- Test with both AWS S3 and local data sources
- Verify GPU acceleration (if available)

## 📄 License

[Specify your license here]

## 🙏 Acknowledgments

-Christopher Matherne 
- FastAPI for the web framework
- All open-source contributors to the dependencies

## 📧 Support

For issues, questions, or contributions, please update the ISSUES thread.

---

**Note**: This is a test implementation designed for demonstration purposes. For production use, additional security, scalability, and compliance measures should be implemented.


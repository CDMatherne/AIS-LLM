"""
Claude-powered conversational agent for AIS fraud detection
"""
import anthropic
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Cache for model detection results (key: API key hash, value: model name)
_model_cache: Dict[str, str] = {}
_model_cache_ttl: Dict[str, datetime] = {}
CACHE_TTL_HOURS = 24  # Cache model detection for 24 hours


def detect_available_claude_model(api_key: str) -> str:
    """
    Detect which Claude model is available with the given API key.
    First attempts to fetch available models from API, then tries in preference order.
    Results are cached to avoid repeated API calls for the same API key.
    
    Returns:
        Model name string
        
    Raises:
        RuntimeError: If no models are available
    """
    # Check cache first (use hash of API key for security)
    import hashlib
    api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
    
    # Check if cached and still valid
    if api_key_hash in _model_cache:
        cache_time = _model_cache_ttl.get(api_key_hash)
        if cache_time and (datetime.now() - cache_time).total_seconds() < (CACHE_TTL_HOURS * 3600):
            logger.info(f"Using cached model: {_model_cache[api_key_hash]}")
            return _model_cache[api_key_hash]
        else:
            # Cache expired, remove it
            _model_cache.pop(api_key_hash, None)
            _model_cache_ttl.pop(api_key_hash, None)
    
    client = anthropic.Anthropic(api_key=api_key)
    
    # Define model preference rankings (higher = better)
    # Sonnet > Haiku (Opus models excluded for cost efficiency)
    model_rankings = {
        "claude-sonnet-4": 1100,       # Claude 4 Sonnet (best available)
        "claude-haiku-4": 1000,        # Claude 4 Haiku
        "claude-3-5-sonnet": 900,      # Claude 3.5 Sonnet
        "claude-3-5-haiku": 850,       # Claude 3.5 Haiku
        "claude-3-sonnet": 700,        # Claude 3 Sonnet
        "claude-3-haiku": 600,         # Claude 3 Haiku
        # Opus models excluded: claude-opus-4, claude-3-5-opus, claude-3-opus
    }
    
    # Fallback list if API query fails (Opus models excluded for cost efficiency)
    fallback_models = [
        ("claude-sonnet-4-5-20250929", "Claude Sonnet 4.5 (Latest)"),
        ("claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet (Oct 2024)"),
        ("claude-3-5-sonnet-20240620", "Claude 3.5 Sonnet (Jun 2024)"),
        ("claude-3-5-haiku-20241022", "Claude 3.5 Haiku (Oct 2024)"),
        ("claude-3-sonnet-20240229", "Claude 3 Sonnet"),
        ("claude-3-haiku-20240307", "Claude 3 Haiku"),
        # Opus models excluded: claude-3-opus-20240229
    ]
    
    logger.info("Detecting available Claude models...")
    
    # Try to get models list from API
    models_to_try = []
    try:
        # Anthropic's models endpoint
        import requests
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }
        response = requests.get(
            "https://api.anthropic.com/v1/models",
            headers=headers,
            timeout=5
        )
        
        if response.status_code == 200:
            models_data = response.json()
            available_models = []
            
            if "data" in models_data:
                for model_info in models_data["data"]:
                    model_id = model_info.get("id", "")
                    # Only include Claude models, exclude Opus models for cost efficiency
                    if model_id.startswith("claude-") and "opus" not in model_id.lower():
                        available_models.append(model_id)
                
                # Sort by preference ranking
                def get_model_score(model_id):
                    for prefix, score in model_rankings.items():
                        if model_id.startswith(prefix):
                            # Add date component for tie-breaking (newer = higher)
                            if "-202" in model_id:  # Has date like -20241022
                                date_part = model_id.split("-202")[-1][:8]
                                try:
                                    date_score = int(date_part) / 100000000  # Normalize to 0-1 range
                                    return score + date_score
                                except:
                                    pass
                            return score
                    return 0
                
                available_models.sort(key=get_model_score, reverse=True)
                
                # Format as tuples with friendly names
                for model_id in available_models:
                    friendly_name = model_id.replace("-", " ").title()
                    models_to_try.append((model_id, friendly_name))
                
                logger.info(f"Found {len(models_to_try)} models from API")
        
    except Exception as e:
        logger.debug(f"Could not fetch models from API: {e}")
    
    # Fall back to hardcoded list if API query failed
    if not models_to_try:
        logger.info("Using fallback model list")
        models_to_try = fallback_models
    
    logger.info(f"Testing {len(models_to_try)} models in preference order...")
    
    for model_id, model_name in models_to_try:
        # CRITICAL: Skip Opus models (cost efficiency)
        if "opus" in model_id.lower():
            logger.debug(f"✗ Skipping Opus model for cost efficiency: {model_name} ({model_id})")
            continue
            
        try:
            # Try a minimal test message
            response = client.messages.create(
                model=model_id,
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}]
            )
            logger.info(f"✓ Successfully connected to {model_name} ({model_id})")
            
            # Cache the result
            _model_cache[api_key_hash] = model_id
            _model_cache_ttl[api_key_hash] = datetime.now()
            
            return model_id
            
        except anthropic.NotFoundError:
            logger.debug(f"✗ {model_name} not available with this API key")
            continue
        except anthropic.PermissionDeniedError:
            logger.debug(f"✗ {model_name} access denied with this API key")
            continue
        except Exception as e:
            # Other errors (rate limit, auth issues, etc.) should be raised
            if "insufficient" in str(e).lower() or "quota" in str(e).lower():
                logger.warning(f"⚠ {model_name} available but quota exceeded")
                continue
            else:
                # Unexpected error - log it but continue trying
                logger.warning(f"⚠ Error testing {model_name}: {str(e)[:100]}")
                continue
    
    # No models available
    error_msg = (
        "No Claude models are available with this API key. "
        "Please check your Anthropic Console (https://console.anthropic.com/) "
        "to verify your account status and model access."
    )
    logger.error(error_msg)
    raise RuntimeError(error_msg)


class AISFraudDetectionAgent:
    """
    Claude-powered agent that understands AIS data, maritime fraud patterns,
    and assists law enforcement in targeting investigations.
    """
    
    def __init__(self, api_key: str, auto_detect_model: bool = True):
        self.client = anthropic.Anthropic(api_key=api_key)
        
        # Auto-detect best available model or use default
        if auto_detect_model:
            try:
                self.model = detect_available_claude_model(api_key)
                logger.info(f"Using Claude model: {self.model}")
            except RuntimeError as e:
                logger.error(f"Model detection failed: {e}")
                raise
        else:
            # Fallback to Haiku if auto-detect is disabled
            self.model = "claude-3-haiku-20240307"
            logger.info(f"Using default Claude model: {self.model}")
        
        self.conversation_history = []
        
        # System prompt with AIS domain knowledge
        self.system_prompt = self._build_system_prompt()
        
        # Tool definitions for Claude function calling
        self.tools = self._define_tools()
    
    def _build_system_prompt(self) -> str:
        """Build comprehensive system prompt with AIS expertise"""
        return """You are an expert AIS (Automatic Identification System) fraud detection assistant 
designed to help law enforcement identify and investigate maritime fraud.

# CRITICAL: System Configuration and Data Access
✅ **YOU HAVE DIRECT ACCESS TO AIS DATA**
- The system is ALREADY CONFIGURED with a data source (AWS S3 or local storage)
- Available data range: October 15, 2024 through March 30, 2025
- You can IMMEDIATELY analyze this data by calling the analysis tools
- **DO NOT** tell users to download or upload data - you already have access!
- **DO NOT** say "I can't access S3" - the backend handles data access for you!
- When users ask about the data, PROCEED DIRECTLY to analysis after setting timespan and vessel types

⚠️ THIS IS A TEST IMPLEMENTATION WITH LIMITED DATA
- You can ONLY answer questions and perform analysis within October 15, 2024 - March 30, 2025
- If a user asks about dates outside this range, politely inform them of the limitation
- Always suggest dates within the valid range when helping users

# Your Capabilities
- Analyze AIS data for suspicious patterns and anomalies (YOU HAVE THE DATA!)
- Understand maritime geography, shipping routes, and jurisdictions
- Define and manage geographic zones (polygons, circles, rectangles, ovals)
- Query historical vessel movements and behavior (within Oct 15, 2024 - Mar 30, 2025)
- Identify vessels of interest by MMSI or behavior patterns
- Provide investigative recommendations based on anomaly patterns
- **IMPORTANT:** All data is accessible through backend tools - no user uploads needed!

# Data Access
- Backend is configured with AWS S3 or local files
- **File Naming Convention:** Files are named by date in format `ais-YYYY-MM-DD.parquet` or `ais-YYYY-MM-DD.csv`
  - Example: `ais-2024-10-15.parquet` contains data for October 15, 2024
  - Alternative format also supported: `YYYY-MM-DD.parquet` (without 'ais-' prefix)
  - When user requests data for a specific date, the system searches for the corresponding file
- Data format: Parquet files (.parquet) or CSV files (.csv) containing AIS records
- Required fields: MMSI, LAT, LON, BaseDateTime, SOG, COG, Heading, VesselType
- You access data by calling analysis tools (run_anomaly_analysis, get_vessel_history, etc.)
- **IMPORTANT:** When a user requests analysis for a date (e.g., "October 15, 2024"), the system automatically searches for and loads the file `ais-2024-10-15.parquet` (or `.csv`)

# AIS Domain Knowledge

**What is AIS?**
Automatic Identification System (AIS) is a communications system using VHF maritime bands to exchange 
navigation data. It provides:
- Dynamic info: position, heading, speed, rate of turn (from onboard sensors)
- Static info: ship name, cargo, destination
- Each station has a unique Maritime Mobile Service Identity (MMSI)

**Legal Requirements:**
- Mandatory under SOLAS for commercial vessels
- Class A transponders (SOLAS vessels): broadcast every 2-10 seconds underway
- Class B transponders (non-SOLAS): broadcast every 30-180 seconds
- Some vessels may turn off AIS illegally (dark shipping)

# Anomaly Types You Detect

1. **AIS Beacon Off**: Vessel disappears for extended periods (>6 hours default)
   - Possible dark shipping, operating in restricted areas
   - May indicate illegal fishing, sanctions evasion, smuggling

2. **AIS Beacon On**: Vessel reappears after being off
   - Position jumps, impossible travel distances
   - May indicate spoofing or manipulation

3. **Speed Anomalies**: Impossible travel distances between positions
   - Exceeds max threshold (550 nm default)
   - Indicates data error, spoofing, or beacon manipulation

4. **COG/Heading Inconsistency**: Large difference between Course Over Ground and Heading
   - Difference exceeds threshold (45° default)
   - May indicate vessel drifting, being towed, or spoofing

5. **Loitering**: Extended presence in small area
   - Remains within radius for extended time
   - May indicate fishing, waiting for rendezvous, or suspicious activity

6. **Rendezvous**: Multiple vessels meeting at sea
   - Potential contraband transfer, illegal transshipment
   - Common in illegal fishing and smuggling operations

7. **Identity Spoofing**: Multiple vessels using same MMSI
   - Fraudulent identity usage
   - Indicates intentional deception

8. **Zone Violations**: Entry into restricted or monitored areas
   - Territorial violations, EEZ incursions
   - Entry to sanctioned ports or exclusion zones

# Interactive Workflow - PRIMARY METHOD

**CRITICAL: When a user requests analysis, follow this EXACT interactive workflow:**

## STEP 1: REQUEST DATE SELECTION (ALWAYS FIRST)
1. Use `request_date_selection` tool to show a date picker popup
   - Default: Start 2024-10-15, End 2024-10-17
   - Message: "Please select the date range for analysis"
2. Wait for user to select dates via popup
3. Once user selects dates, use `set_analysis_timespan` to store them
4. Confirm: "✅ Date range set: [START] to [END]"

## STEP 2: REQUEST VESSEL TYPE SELECTION
1. Use `request_vessel_type_selection` tool to show vessel type checkboxes
   - Default: ["Cargo"] (VesselType 70)
   - Message: "Please select vessel types to analyze"
2. **IMPORTANT:** The user can respond in TWO ways:
   - **Via popup:** Wait for user to select vessel types via popup (preferred)
   - **Via text:** If user provides direct text answer (e.g., "type 70", "Cargo", "use Cargo"), accept it and proceed without waiting for popup
3. If user provides direct text answer, extract vessel type(s) from their message and proceed to STEP 3
4. Confirm: "✅ Selected vessel types: [TYPES]"

## STEP 3: REQUEST ANOMALY TYPE SELECTION
1. Use `request_anomaly_selection` tool to show anomaly type checkboxes
   - Default: ["ais_beacon_on", "ais_beacon_off", "excessive_travel_distance_fast", "cog-heading_inconsistency"]
   - Exclude: "loitering" and "rendezvous" by default
   - Message: "Please select anomaly types to detect"
2. Wait for user to select anomaly types via popup
3. Confirm: "✅ Selected anomaly types: [TYPES]"

## STEP 4: REQUEST OUTPUT SELECTION
1. Use `request_output_selection` tool to show output format checkboxes
   - Default: ["consolidated_events_csv", "heatmap", "event_map"]
   - Message: "Please select output formats to generate"
2. Wait for user to select outputs via popup
3. Confirm: "✅ Selected outputs: [OUTPUTS]"

## STEP 5: RUN ANALYSIS
1. Use `run_anomaly_analysis` with the collected parameters:
   - start_date, end_date (from Step 1)
   - vessel_types (from Step 2)
   - anomaly_types (from Step 3)
2. **CRITICAL: Communicate progress updates to the user**
   - When the tool result includes a `progress_updates` field, read through them and inform the user of the current status
   - Format progress updates as: "⏳ [STAGE]: [MESSAGE]"
   - Examples: "⏳ [SEARCHING] Searching for data files...", "⏳ [LOADING] Loading data from 3 files...", "⏳ [ANALYZING] Starting anomaly detection..."
   - Send these updates to the user immediately so they know the analysis is progressing
3. Once complete, generate the requested outputs from Step 4

## STEP 6: SAVE OUTPUTS WITH TIMESTAMPS
1. **CRITICAL: All outputs must be tagged with date/time to prevent overwrites**
2. Format: `filename_YYYYMMDD_HHMMSS.ext`
3. Example: `consolidated_events_20241111_163045.csv`
4. For each output format:
   - **consolidated_events_csv**: Use `export_to_csv` with export_type="anomalies"
   - **heatmap**: Use `create_anomaly_heatmap`
   - **event_map**: Use `create_all_anomalies_map`
   - **statistics_csv**: Use `export_to_csv` with export_type="statistics"
   - **excel_report**: Use `export_to_excel`
   - **vessel_tracks**: Use `create_vessel_track_map` for each vessel
5. Inform user: "✅ Analysis complete! Files saved to output folder with timestamp [TIMESTAMP]. All outputs are timestamped to prevent overwrites."

## STEP 7: CONTINUED INTERACTION
- After analysis completes, allow user to ask questions about the results
- If user requests data outside original scope (different vessel types, anomalies, dates):
  - Warn: "⚠️ This requires a new analysis with different parameters. This will create new timestamped files and will not overwrite existing outputs."
  - Ask: "Would you like me to run a new analysis?"
  - If yes, start workflow from STEP 1 again
  - **NEVER overwrite existing outputs** - always create new timestamped files


# Geographic Capabilities & Water Bodies Knowledge 🌊
You have comprehensive knowledge of 24 major water bodies worldwide:

**Oceans (5):** Pacific, Atlantic, Indian, Arctic, Southern
**Seas (11):** 
- Atlantic Region: Caribbean, Mediterranean, Black Sea, North Sea, Baltic Sea
- Pacific Region: South China Sea, East China Sea, Sea of Japan, Bering Sea
- Indian Ocean: Arabian Sea, Red Sea
**Gulfs & Bays (3):** Gulf of Mexico, Persian Gulf, Bay of Bengal
**Strategic Waterways (5):** Strait of Gibraltar, Strait of Malacca, Strait of Hormuz, Suez Canal, Panama Canal

You can:
- Identify which water body any vessel or coordinate is in
- Filter analyses by specific water bodies
- List all available water bodies for users
- Work with lat/lon coordinates (decimal degrees)
- Create custom polygons, circles, ovals, rectangles
- Use drawn polygons from the map interface
- Understand maritime boundaries (EEZ, territorial waters)

# Vessel Type Filtering Capabilities 🚢
You have comprehensive knowledge of 58 AIS vessel types (codes 20-99) organized into 8 categories:

**Commercial Vessels:**
- **Cargo (70-79):** Container ships, bulk carriers, general cargo
  - Code 70: All cargo vessels
  - Codes 71-74: Hazardous cargo (categories A-D)
- **Tanker (80-89):** Oil tankers, chemical tankers, LNG carriers
  - Code 80: All tankers
  - Codes 81-84: Hazardous tankers (categories A-D)

**Special Purpose (50-59):** 
- Code 50: Pilot Vessel
- Code 51: Search and Rescue (SAR)
- Code 52: Tug
- Code 55: **Law Enforcement** (friendly assets)
- Code 58: Medical Transport

**Special (30-39):**
- Code 30: **Fishing** (illegal fishing monitoring)
- Code 35: Military ops
- Code 36-37: Sailing, Pleasure Craft

**Other Categories:**
- Passenger (60-69): Cruise ships, ferries
- High Speed Craft/HSC (40-49): Fast ferries
- Wing in Ground/WIG (20-29): Ground-effect vehicles
- Other (90-99): Miscellaneous types

**Hazardous Cargo:** Any code ending in 1-4 carries dangerous goods
- Category A (1): Major hazard (flammable/toxic gases)
- Category B (2): Medium hazard
- Category C (3): Minor hazard
- Category D (4): Recognizable hazard

**When to use vessel type filtering:**
- Illegal fishing investigations → Filter to code 30
- Sanctions monitoring → Filter to tankers (80-89) and cargo (70-79)
- Hazmat compliance → Filter to codes ending in 1-4
- Identify friendly assets → Filter to code 55 (Law Enforcement)
- Reduce noise in analysis → Filter to relevant vessel categories

# Map & Visualization Outputs 🗺️
You can generate professional interactive maps and visualizations:

**Built-in Map Generators:**
- create_all_anomalies_map: Interactive map showing all detected anomalies with color-coded markers, clustering, popups with vessel details, and optional heatmap/grid overlay
- create_vessel_track_map: Track map for specific vessel showing complete movement history with anomalies highlighted
- create_anomaly_heatmap: Heatmap showing geographic density of anomalies - ideal for identifying hotspots

**Built-in Chart Generators:**
- create_anomaly_types_chart: Bar or pie chart showing distribution of anomaly types
- create_top_vessels_chart: Horizontal bar chart of vessels with most anomalies
- create_anomalies_by_date_chart: Time series showing anomalies over time
- create_3d_bar_chart: 3D visualization of anomalies by type and date
- create_scatterplot: Interactive geographic scatterplot on world map

**Export Capabilities:**
- export_to_csv: Export anomalies summary or statistics to CSV
- export_to_excel: Comprehensive Excel report with multiple sheets
- list_available_exports: See all generated files

**Maps include:**
- Multiple base map options (street, terrain, satellite)
- Layer controls for toggling visibility
- Fullscreen mode and measurement tools
- Legend showing anomaly types with counts
- Clustering for performance with large datasets
- Optional lat/long grid overlay for precise navigation

# Dynamic Visualization Creation ✨
You have the POWERFUL ability to create custom visualizations programmatically:
- When standard outputs don't meet user needs, you can WRITE PYTHON CODE to create new visualizations
- Use matplotlib for static charts, plotly for interactive plots, folium for maps
- Access code templates with get_visualization_templates
- Your code should define a generate_visualization(data, parameters, output_dir) function
- If a visualization works well, SAVE IT TO THE REGISTRY to make it available for all users
- You can reuse visualizations created by other users - check list_custom_visualizations

Example: User wants "a 3D scatter plot showing vessel speed vs time colored by anomaly type"
→ You write Python code using plotly to create exactly that
→ If successful and useful, save it so other users can use it too

This makes the system self-improving - every good visualization adds to the toolkit!

# Communication Style
- Professional and authoritative
- Clear, concise explanations
- Use maritime and law enforcement terminology appropriately
- Provide context and rationale for recommendations
- Be transparent about limitations and uncertainties"""

    def _define_tools(self) -> List[Dict[str, Any]]:
        """Define tools/functions Claude can call"""
        return [
            {
                "name": "run_anomaly_analysis",
                "description": "Execute AIS anomaly detection on specified date range and geographic area",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "start_date": {
                            "type": "string", 
                            "description": "Start date in YYYY-MM-DD format (must be between 2024-10-15 and 2025-03-30)"
                        },
                        "end_date": {
                            "type": "string", 
                            "description": "End date in YYYY-MM-DD format (must be between 2024-10-15 and 2025-03-30)"
                        },
                        "geographic_zone": {
                            "type": "object",
                            "description": "Geographic area to analyze (optional - if omitted, analyzes all data)",
                            "properties": {
                                "type": {"type": "string", "enum": ["polygon", "rectangle", "circle", "oval"]},
                                "coordinates": {"type": "array", "description": "Array of [lon, lat] points for polygon/rectangle, or [center_lon, center_lat] for circle/oval"},
                                "radius_nm": {"type": "number", "description": "Radius in nautical miles (for circle)"},
                                "major_axis_nm": {"type": "number", "description": "Major axis in nautical miles (for oval)"},
                                "minor_axis_nm": {"type": "number", "description": "Minor axis in nautical miles (for oval)"}
                            }
                        },
                        "anomaly_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific anomaly types to detect (e.g., ['ais_beacon_off', 'zone_violations']). If omitted, detects all types."
                        },
                        "mmsi_filter": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific vessel MMSIs to investigate (optional - if omitted, analyzes all vessels)"
                        }
                    },
                    "required": ["start_date", "end_date"]
                }
            },
            {
                "name": "get_vessel_history",
                "description": "Retrieve complete movement history and track for specific vessel(s) by MMSI",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "mmsi": {
                            "type": "array", 
                            "items": {"type": "string"},
                            "description": "List of vessel MMSI numbers to look up"
                        },
                        "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                        "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"}
                    },
                    "required": ["mmsi", "start_date", "end_date"]
                }
            },
            {
                "name": "create_geographic_zone",
                "description": "Create a new monitored geographic zone for analysis",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name for this zone"},
                        "zone_type": {
                            "type": "string", 
                            "enum": ["polygon", "rectangle", "circle", "oval"],
                            "description": "Type of geographic zone"
                        },
                        "coordinates": {
                            "type": "array",
                            "description": "Coordinates: array of [lon, lat] for polygon, [min_lon, min_lat, max_lon, max_lat] for rectangle, [center_lon, center_lat] for circle/oval"
                        },
                        "radius_nm": {"type": "number", "description": "Radius in nautical miles (required for circle)"},
                        "major_axis_nm": {"type": "number", "description": "Major axis in nautical miles (required for oval)"},
                        "minor_axis_nm": {"type": "number", "description": "Minor axis in nautical miles (required for oval)"},
                        "rotation": {"type": "number", "description": "Rotation in degrees (optional, for oval)"},
                        "description": {"type": "string", "description": "Description of this zone's purpose"}
                    },
                    "required": ["name", "zone_type", "coordinates"]
                }
            },
            {
                "name": "identify_high_risk_vessels",
                "description": "Find vessels with most anomalies matching specified criteria - useful for prioritizing investigations",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "start_date": {"type": "string"},
                        "end_date": {"type": "string"},
                        "geographic_zone": {"type": "object", "description": "Optional - limit to specific area"},
                        "min_anomalies": {"type": "integer", "description": "Minimum number of anomalies to include vessel"},
                        "top_n": {"type": "integer", "description": "Return top N vessels (default 10)"}
                    },
                    "required": ["start_date", "end_date"]
                }
            },
            {
                "name": "list_all_water_bodies",
                "description": "List all 24 defined water bodies (oceans, seas, gulfs, bays, straits, canals) that the system recognizes. Use this to help users understand which geographic regions are available for analysis.",
                "input_schema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "identify_vessel_location",
                "description": "Identify which water body (ocean, sea, gulf, strait, etc.) a specific location or vessel is in based on lat/lon coordinates",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "longitude": {
                            "type": "number",
                            "description": "Longitude coordinate (-180 to 180)"
                        },
                        "latitude": {
                            "type": "number",
                            "description": "Latitude coordinate (-90 to 90)"
                        },
                        "mmsi": {
                            "type": "string",
                            "description": "Optional: MMSI of vessel to identify location for (if provided, gets current position from latest data)"
                        }
                    }
                }
            },
            {
                "name": "lookup_geographic_region",
                "description": "Convert named maritime region to detailed geographic zone. Supports 24 water bodies: Pacific/Atlantic/Indian/Arctic/Southern Oceans; Mediterranean/Caribbean/South China/East China/Sea of Japan/Bering/Arabian/Red/North/Baltic/Black Seas; Gulf of Mexico/Persian Gulf; Bay of Bengal; Straits of Gibraltar/Malacca/Hormuz; Suez/Panama Canals",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "region_name": {
                            "type": "string",
                            "description": "Name of water body (e.g., 'Mediterranean Sea', 'Strait of Hormuz', 'Panama Canal', 'Pacific Ocean')"
                        }
                    },
                    "required": ["region_name"]
                }
            },
            {
                "name": "generate_investigation_report",
                "description": "Generate formal report for law enforcement use from analysis results",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "analysis_id": {"type": "string", "description": "ID of analysis to report on"},
                        "report_type": {
                            "type": "string", 
                            "enum": ["summary", "detailed", "tactical"],
                            "description": "Type of report: summary (executive), detailed (full analysis), tactical (operational)"
                        },
                        "include_maps": {"type": "boolean", "description": "Include map visualizations"},
                        "include_vessel_details": {"type": "boolean", "description": "Include detailed vessel information"}
                    },
                    "required": ["analysis_id"]
                }
            },
            {
                "name": "export_to_csv",
                "description": "Export analysis results to CSV file for download - creates AIS_Anomalies_Summary.csv or Analysis_Statistics.csv",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "analysis_id": {"type": "string", "description": "ID of analysis to export"},
                        "export_type": {
                            "type": "string",
                            "enum": ["anomalies", "statistics"],
                            "description": "Type of export: 'anomalies' for full anomaly list, 'statistics' for summary stats"
                        }
                    },
                    "required": ["analysis_id", "export_type"]
                }
            },
            {
                "name": "export_to_excel",
                "description": "Export comprehensive analysis statistics to Excel file with multiple sheets - creates Analysis_Statistics.xlsx",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "analysis_id": {"type": "string", "description": "ID of analysis to export"}
                    },
                    "required": ["analysis_id"]
                }
            },
            {
                "name": "list_available_exports",
                "description": "List all available export files that can be downloaded",
                "input_schema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "create_custom_visualization",
                "description": "Create a NEW custom visualization by writing Python code using matplotlib, plotly, or folium. Use this when standard outputs don't meet user needs. Successful visualizations can be saved for all users.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "analysis_id": {
                            "type": "string",
                            "description": "ID of analysis to visualize"
                        },
                        "code": {
                            "type": "string",
                            "description": "Python code that defines a 'generate_visualization(data, parameters, output_dir)' function. The function receives a pandas DataFrame with anomaly data and should return the output file path."
                        },
                        "parameters": {
                            "type": "object",
                            "description": "Parameters for the visualization (e.g., title, filename, colors)"
                        },
                        "name": {
                            "type": "string",
                            "description": "Name for this visualization (required if saving)"
                        },
                        "description": {
                            "type": "string",
                            "description": "Description of what this visualization does"
                        },
                        "save_to_registry": {
                            "type": "boolean",
                            "description": "Whether to save this visualization to make it available for all users"
                        }
                    },
                    "required": ["analysis_id", "code"]
                }
            },
            {
                "name": "execute_saved_visualization",
                "description": "Execute a visualization that has been previously saved to the registry by you or other users",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "analysis_id": {
                            "type": "string",
                            "description": "ID of analysis to visualize"
                        },
                        "viz_id": {
                            "type": "string",
                            "description": "ID of the saved visualization to execute"
                        },
                        "parameters": {
                            "type": "object",
                            "description": "Parameters to customize the visualization"
                        }
                    },
                    "required": ["analysis_id", "viz_id"]
                }
            },
            {
                "name": "list_custom_visualizations",
                "description": "List all available custom visualizations created by users. Shows popular and highly-rated visualizations that can be executed.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "viz_type": {
                            "type": "string",
                            "enum": ["map", "chart", "report", "interactive_chart", "static_chart", "custom"],
                            "description": "Filter by visualization type (optional)"
                        }
                    }
                }
            },
            {
                "name": "get_visualization_templates",
                "description": "Get code templates and examples for creating custom visualizations. Use this to see example code for matplotlib charts, plotly interactive plots, folium maps, and custom reports.",
                "input_schema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "create_all_anomalies_map",
                "description": "Create an interactive Folium map showing all anomalies from an analysis. Markers are color-coded by anomaly type, include vessel details in popups, and can be clustered for performance.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "analysis_id": {
                            "type": "string",
                            "description": "ID of analysis to map"
                        },
                        "show_clustering": {
                            "type": "boolean",
                            "description": "Whether to cluster markers for better performance (default: true)"
                        },
                        "show_heatmap": {
                            "type": "boolean",
                            "description": "Whether to show heatmap layer in addition to markers (default: false)"
                        },
                        "show_grid": {
                            "type": "boolean",
                            "description": "Whether to show lat/long grid overlay (default: false)"
                        }
                    },
                    "required": ["analysis_id"]
                }
            },
            {
                "name": "create_vessel_track_map",
                "description": "Create an interactive map showing a specific vessel's complete movement track with anomalies highlighted. Shows track path, start/end points, and anomaly locations.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "analysis_id": {
                            "type": "string",
                            "description": "ID of analysis containing vessel data"
                        },
                        "mmsi": {
                            "type": "string",
                            "description": "MMSI of the vessel to track"
                        }
                    },
                    "required": ["analysis_id", "mmsi"]
                }
            },
            {
                "name": "create_anomaly_heatmap",
                "description": "Create a heatmap showing density of anomalies across geographic regions. Useful for identifying hotspots of suspicious activity.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "analysis_id": {
                            "type": "string",
                            "description": "ID of analysis to create heatmap from"
                        }
                    },
                    "required": ["analysis_id"]
                }
            },
            {
                "name": "create_anomaly_types_chart",
                "description": "Create a bar or pie chart showing the distribution of anomaly types detected in the analysis. Helps visualize which types of anomalies are most common.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "analysis_id": {
                            "type": "string",
                            "description": "ID of analysis to create chart from"
                        },
                        "chart_type": {
                            "type": "string",
                            "enum": ["bar", "pie"],
                            "description": "Type of chart to create (default: bar)"
                        }
                    },
                    "required": ["analysis_id"]
                }
            },
            {
                "name": "create_top_vessels_chart",
                "description": "Create a horizontal bar chart showing the vessels with the most anomalies. Helps identify high-priority vessels for investigation.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "analysis_id": {
                            "type": "string",
                            "description": "ID of analysis to create chart from"
                        },
                        "top_n": {
                            "type": "integer",
                            "description": "Number of top vessels to show (default: 10)"
                        }
                    },
                    "required": ["analysis_id"]
                }
            },
            {
                "name": "create_anomalies_by_date_chart",
                "description": "Create a time series chart showing how anomalies are distributed over time. Helps identify temporal patterns and trends.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "analysis_id": {
                            "type": "string",
                            "description": "ID of analysis to create chart from"
                        }
                    },
                    "required": ["analysis_id"]
                }
            },
            {
                "name": "create_3d_bar_chart",
                "description": "Create a 3D bar chart showing the distribution of anomalies across both type and time. Provides a comprehensive view of patterns.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "analysis_id": {
                            "type": "string",
                            "description": "ID of analysis to create chart from"
                        }
                    },
                    "required": ["analysis_id"]
                }
            },
            {
                "name": "create_scatterplot",
                "description": "Create an interactive geographic scatterplot showing the distribution of anomalies on a world map. Color-coded by anomaly type with hover details.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "analysis_id": {
                            "type": "string",
                            "description": "ID of analysis to create scatterplot from"
                        }
                    },
                    "required": ["analysis_id"]
                }
            },
            {
                "name": "list_vessel_types",
                "description": "List all 58 vessel type codes organized by category (Cargo, Tanker, Fishing, Law Enforcement, Passenger, etc.). Use this to help users understand which vessel types can be filtered in analysis.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": ["WIG", "Special", "HSC", "Special Purpose", "Passenger", "Cargo", "Tanker", "Other", "all"],
                            "description": "Optionally filter to specific category (default: all)"
                        }
                    }
                }
            },
            {
                "name": "get_vessel_type_info",
                "description": "Get detailed information about a specific AIS vessel type code",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "vessel_type_code": {
                            "type": "integer",
                            "description": "AIS vessel type code (20-99)"
                        }
                    },
                    "required": ["vessel_type_code"]
                }
            },
            {
                "name": "request_date_selection",
                "description": "Trigger a date picker popup dialog for the user to select start and end dates. Use this at the beginning of analysis workflow to collect date range from user.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Message to display to user in the popup"
                        },
                        "default_start": {
                            "type": "string",
                            "description": "Default start date (YYYY-MM-DD, default: 2024-10-15)"
                        },
                        "default_end": {
                            "type": "string",
                            "description": "Default end date (YYYY-MM-DD, default: 2024-10-17)"
                        },
                        "min_date": {
                            "type": "string",
                            "description": "Minimum selectable date (YYYY-MM-DD, default: 2024-10-15)"
                        },
                        "max_date": {
                            "type": "string",
                            "description": "Maximum selectable date (YYYY-MM-DD, default: 2025-03-30)"
                        }
                    }
                }
            },
            {
                "name": "request_vessel_type_selection",
                "description": "Trigger a vessel type selection popup with checkboxes. Use this to collect vessel type preferences from the user before running analysis.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Message to display to user in the popup"
                        },
                        "default_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Default selected vessel types (default: ['Cargo'])"
                        },
                        "allow_multiple": {
                            "type": "boolean",
                            "description": "Whether to allow multiple selections (default: true)"
                        }
                    }
                }
            },
            {
                "name": "request_anomaly_selection",
                "description": "Trigger an anomaly type selection popup with checkboxes. Use this to collect which anomaly types the user wants to detect.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Message to display to user in the popup"
                        },
                        "default_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Default selected anomaly types (default: excludes loitering and rendezvous)"
                        },
                        "allow_multiple": {
                            "type": "boolean",
                            "description": "Whether to allow multiple selections (default: true)"
                        }
                    }
                }
            },
            {
                "name": "request_output_selection",
                "description": "Trigger an output format selection popup with checkboxes. Use this to collect which outputs the user wants generated (CSV, maps, charts, etc.).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Message to display to user in the popup"
                        },
                        "default_outputs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Default selected outputs (default: ['consolidated_events_csv', 'heatmap', 'event_map'])"
                        },
                        "allow_multiple": {
                            "type": "boolean",
                            "description": "Whether to allow multiple selections (default: true)"
                        }
                    }
                }
            },
            {
                "name": "set_analysis_timespan",
                "description": "Set or update the date range for analysis. Users MUST set this before running any anomaly analysis. This helps focus the investigation on a specific time period.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "start_date": {
                            "type": "string",
                            "description": "Start date in YYYY-MM-DD format (must be between 2024-10-15 and 2025-03-30)"
                        },
                        "end_date": {
                            "type": "string",
                            "description": "End date in YYYY-MM-DD format (must be between 2024-10-15 and 2025-03-30)"
                        }
                    },
                    "required": ["start_date", "end_date"]
                }
            },
            {
                "name": "get_current_timespan",
                "description": "Get the currently set date range for analysis. Check this to see what timespan the user has selected.",
                "input_schema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
    
    def _validate_conversation_history(self):
        """
        Validate conversation history to ensure no orphaned tool_use blocks.
        Removes any assistant messages with tool_use that don't have corresponding tool_result.
        Handles both dict format and Anthropic ContentBlock format.
        Optimized to only check recent messages for better performance.
        """
        if len(self.conversation_history) < 1:
            return
        
        # Only validate last 20 messages for performance (orphaned tool_use blocks are typically recent)
        # This significantly reduces processing time for long conversations
        check_window = 20
        start_index = max(0, len(self.conversation_history) - check_window)
        
        # Track tool_use IDs that need tool_result (map: tool_use_id -> message_index)
        pending_tool_uses = {}
        
        # Helper function to extract tool_use_id from a content block
        def get_tool_use_id(block):
            if isinstance(block, dict):
                if block.get("type") == "tool_use":
                    return block.get("id")
            elif hasattr(block, 'type') and block.type == "tool_use":
                return block.id if hasattr(block, 'id') else None
            return None
        
        # Helper function to extract tool_use_id from a tool_result block
        def get_tool_result_id(block):
            if isinstance(block, dict):
                if block.get("type") == "tool_result":
                    return block.get("tool_use_id")
            elif hasattr(block, 'type') and block.type == "tool_result":
                return block.tool_use_id if hasattr(block, 'tool_use_id') else None
            return None
        
        # Scan through recent messages to find orphaned tool_use blocks
        i = start_index
        while i < len(self.conversation_history):
            msg = self.conversation_history[i]
            
            if msg["role"] == "assistant":
                # Check if this assistant message has tool_use blocks
                tool_use_ids_in_msg = []
                if isinstance(msg["content"], list):
                    for block in msg["content"]:
                        tool_use_id = get_tool_use_id(block)
                        if tool_use_id:
                            tool_use_ids_in_msg.append(tool_use_id)
                            pending_tool_uses[tool_use_id] = i
                
                # Check if the immediately following message has tool_result for these tool_uses
                if tool_use_ids_in_msg and i + 1 < len(self.conversation_history):
                    next_msg = self.conversation_history[i + 1]
                    if next_msg["role"] == "user":
                        if isinstance(next_msg["content"], list):
                            for block in next_msg["content"]:
                                tool_result_id = get_tool_result_id(block)
                                if tool_result_id and tool_result_id in tool_use_ids_in_msg:
                                    pending_tool_uses.pop(tool_result_id, None)
            
            elif msg["role"] == "user":
                # Check if this user message has tool_result blocks that resolve pending tool_uses
                if isinstance(msg["content"], list):
                    for block in msg["content"]:
                        tool_result_id = get_tool_result_id(block)
                        if tool_result_id:
                            pending_tool_uses.pop(tool_result_id, None)
            
            i += 1
        
        # If there are pending tool_uses, remove the assistant messages that contain them
        if pending_tool_uses:
            logger.warning(f"Found {len(pending_tool_uses)} orphaned tool_use block(s), cleaning conversation history")
            # Remove messages from the end (reverse order to maintain indices)
            indices_to_remove = sorted(set(pending_tool_uses.values()), reverse=True)
            for idx in indices_to_remove:
                logger.warning(f"Removing orphaned tool_use block from message {idx}")
                self.conversation_history.pop(idx)
    
    async def chat(self, user_message: str, session_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user message and return Claude's response with any tool calls
        
        Args:
            user_message: User's message
            session_context: Context including map state, zones, etc.
        
        Returns:
            Dict with message, tool_calls, and metadata
        """
        # Validate history before adding new message
        self._validate_conversation_history()
        
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        try:
            # Limit conversation history to last 50 messages to reduce token usage and improve speed
            # Keep recent context but prevent excessive growth
            max_history_messages = 50
            if len(self.conversation_history) > max_history_messages:
                # Keep first message (usually important context) and last N messages
                trimmed_history = [self.conversation_history[0]] + self.conversation_history[-max_history_messages+1:]
                logger.debug(f"Trimmed conversation history from {len(self.conversation_history)} to {len(trimmed_history)} messages")
                messages_to_send = trimmed_history
            else:
                messages_to_send = self.conversation_history
            
            # Call Claude with tools
            # Reduced max_tokens from 4096 to 2048 for faster responses (still plenty for most interactions)
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,  # Reduced for faster responses
                system=self.system_prompt,
                messages=messages_to_send,
                tools=self.tools,
                temperature=0.5  # Reduced from 0.7 for faster, more deterministic responses
            )
            
            # Process response and tool calls
            result = {
                "message": "",
                "tool_calls": [],
                "stop_reason": response.stop_reason
            }
            
            for content_block in response.content:
                if content_block.type == "text":
                    result["message"] += content_block.text
                elif content_block.type == "tool_use":
                    result["tool_calls"].append({
                        "id": content_block.id,
                        "name": content_block.name,
                        "input": content_block.input
                    })
            
            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": response.content
            })
            
            logger.info(f"Claude response generated with {len(result['tool_calls'])} tool calls")
            return result
            
        except Exception as e:
            logger.error(f"Error in Claude chat: {e}")
            
            # If error occurred, remove the user message we just added to avoid orphaned tool_use
            if self.conversation_history and self.conversation_history[-1]["role"] == "user":
                self.conversation_history.pop()
            
            return {
                "message": f"I apologize, but I encountered an error: {str(e)}. Please try again.",
                "tool_calls": [],
                "error": str(e)
            }
    
    async def process_tool_results(self, tool_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Send tool execution results back to Claude for interpretation
        
        Args:
            tool_results: List of tool results with tool_call_id and result
        
        Returns:
            Claude's interpretation of the results
        """
        # Validate history before adding tool results - this removes orphaned tool_use blocks
        self._validate_conversation_history()
        
        # Get all valid tool_use_ids that exist in the conversation history
        valid_tool_use_ids = set()
        for msg in self.conversation_history:
            if msg["role"] == "assistant":
                if isinstance(msg["content"], list):
                    for block in msg["content"]:
                        tool_use_id = None
                        if isinstance(block, dict):
                            if block.get("type") == "tool_use":
                                tool_use_id = block.get("id")
                        elif hasattr(block, 'type') and block.type == "tool_use":
                            tool_use_id = block.id if hasattr(block, 'id') else None
                        
                        if tool_use_id:
                            valid_tool_use_ids.add(tool_use_id)
        
        # Filter out tool results that correspond to removed tool_use blocks
        filtered_tool_results = [
            result for result in tool_results 
            if result["tool_call_id"] in valid_tool_use_ids
        ]
        
        # If no valid tool results remain, return early
        if not filtered_tool_results:
            logger.warning(f"All {len(tool_results)} tool result(s) were filtered out - corresponding tool_use blocks were removed")
            return {
                "message": "The previous tool calls were invalidated. Please try again.",
                "tool_calls": [],
                "error": "Tool results filtered - orphaned tool_use blocks removed"
            }
        
        # Check if any tool results contain progress updates and add them as text before tool results
        progress_summary = []
        for result in filtered_tool_results:
            tool_result = result.get("result", {})
            if isinstance(tool_result, dict) and "progress_updates" in tool_result:
                progress_updates = tool_result["progress_updates"]
                if progress_updates:
                    tool_name = result.get("tool_name", "analysis")
                    progress_summary.append(f"\nProgress updates from {tool_name}:")
                    for update in progress_updates:
                        stage = update.get('stage', 'unknown')
                        message = update.get('message', '')
                        progress_summary.append(f"  [{stage}]: {message}")
        
        # Add only valid tool results to conversation
        tool_result_content = []
        for result in filtered_tool_results:
            tool_result_content.append({
                "type": "tool_result",
                "tool_use_id": result["tool_call_id"],
                "content": json.dumps(result["result"], default=str)
            })
        
        # Prepend progress summary as text if available
        if progress_summary:
            progress_text = "\n".join(progress_summary)
            tool_result_content.insert(0, progress_text)
        
        self.conversation_history.append({
            "role": "user",
            "content": tool_result_content
        })
        
        try:
            # Limit conversation history to last 50 messages
            max_history_messages = 50
            if len(self.conversation_history) > max_history_messages:
                trimmed_history = [self.conversation_history[0]] + self.conversation_history[-max_history_messages+1:]
                logger.debug(f"Trimmed conversation history from {len(self.conversation_history)} to {len(trimmed_history)} messages")
                messages_to_send = trimmed_history
            else:
                messages_to_send = self.conversation_history
            
            # Get Claude's interpretation
            # Reduced max_tokens for faster responses
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,  # Reduced for faster responses
                system=self.system_prompt,
                messages=messages_to_send,
                tools=self.tools,
                temperature=0.5  # Reduced for faster, more deterministic responses
            )
            
            result = {"message": "", "tool_calls": []}
            for content_block in response.content:
                if content_block.type == "text":
                    result["message"] += content_block.text
                elif content_block.type == "tool_use":
                    result["tool_calls"].append({
                        "id": content_block.id,
                        "name": content_block.name,
                        "input": content_block.input
                    })
            
            self.conversation_history.append({
                "role": "assistant",
                "content": response.content
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing tool results: {e}")
            
            # If error occurred, remove the tool_result message we just added
            if self.conversation_history and self.conversation_history[-1]["role"] == "user":
                self.conversation_history.pop()
            
            return {
                "message": f"I encountered an error interpreting the results: {str(e)}",
                "tool_calls": [],
                "error": str(e)
            }
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
        logger.info("Conversation history reset")
    
    def get_conversation_length(self) -> int:
        """Get number of messages in conversation"""
        return len(self.conversation_history)


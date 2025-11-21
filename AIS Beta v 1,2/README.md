# AIS Shipping Fraud Detection System

**Version:** 1.2 Beta  
**Team:** Dreadnaught  
**Author:** Chris Matherne  
**Event:** Datathon 2025

## Overview

The AIS Shipping Fraud Detection System analyzes Automatic Identification System (AIS) data to detect potentially fraudulent shipping activities. It identifies anomalies such as vessels turning off their AIS beacons (sudden disappearances), sudden reappearances, unusual travel distances, and large inconsistencies between reported Course Over Ground (COG) and Heading.

**Note:** In this test implementation, only data from October 2024 to March 2025 is available for use.

---

## Table of Contents

1. [Installation](#installation)
2. [Graphical User Interface (SFD_GUI.py)](#graphical-user-interface-sfd_guipy)
3. [Command-Line Interface (SFD.py)](#command-line-interface-sfdpy)
4. [Configuration File (config.ini)](#configuration-file-configini)
5. [Graphics and Icon Files](#graphics-and-icon-files)
6. [Anomaly Types Explained](#anomaly-types-explained)
7. [Output Files](#output-files)
8. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Python 3.8 or higher (Python 3.14 fully supported)
- Windows, Linux, or macOS

### Step 1: Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

**Note for Linux users:** You may need to install tkinter separately:
```bash
sudo apt-get install python3-tk
```

### Step 2: Required Files

Ensure the following files are in the same directory as the scripts:

- **SFD_GUI.py** - Main graphical user interface
- **SFD.py** - Command-line analysis engine
- **config.ini** - Configuration file (created automatically if missing)
- **SFD_AI_banner.png** - Banner image for GUI (optional)
- **SFDLoad.png** - Loading screen image (optional)
- **SFD.ico** - Application icon (optional)

### Step 3: GPU Support (Optional)

For GPU acceleration:

**NVIDIA GPUs:**
- Install CUDA toolkit
- Packages will be installed via requirements.txt (cudf, cupy, cuml)

**AMD GPUs:**
- Install AMD HIP SDK from: https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html
- Install cupy-rocm: `pip install cupy-rocm`
- Or install PyHIP: `pip install pyhip`

---

## Graphical User Interface (SFD_GUI.py)

The GUI provides an intuitive interface for configuring and running analyses. Launch it with:

```bash
python SFD_GUI.py
```

### GUI Tabs Overview

The GUI consists of 7 main tabs:

1. **Startup** - Date selection and main controls
2. **Ship Types** - Select vessel types to analyze
3. **Anomaly Types** - Choose anomaly types and set thresholds
4. **Analysis Filters** - Geographic, time, and vessel filters
5. **Data** - Data source and processing options
6. **Zone Violations** - Manage restricted zones
7. **Output Controls** - Configure report outputs

---

### Tab 1: Startup

**Purpose:** Set the analysis date range and execute the analysis.

#### Controls:

- **Run Analysis** (Green Button)
  - Executes the analysis with current settings
  - Shows progress in a popup window
  - Results are saved to the output directory

- **Save Configuration**
  - Saves all current settings to `config.ini`
  - Settings persist between sessions

- **Check for GPU Acceleration**
  - Tests GPU availability
  - Shows GPU status and installation options

- **Exit** (Red Button)
  - Closes the application

#### Date Range Selection:

- **Start Date** - Beginning of analysis period (format: YYYY-MM-DD)
- **End Date** - End of analysis period (format: YYYY-MM-DD)

**Note:** If `tkcalendar` is installed, you'll see calendar widgets. Otherwise, use text entry fields.

#### Description:

The tab includes a description explaining the system's purpose and available data range.

---

### Tab 2: Ship Types

**Purpose:** Select which vessel types to include in the analysis.

#### Controls:

- **Category Buttons:**
  - **Select All** - Selects all ship types
  - **Deselect All** - Deselects all ship types
  - Category-specific buttons (Cargo, Tanker, etc.) - Select/deselect by category

#### Ship Type Categories:

- **WIG** (Wing in ground) - Types 20-29
- **HSC** (High speed craft) - Types 40-49
- **Special Purpose** - Types 50-59
- **Passenger** - Types 60-69
- **Cargo** - Types 70-79
- **Tanker** - Types 80-89
- **Other** - Types 90-99

Each ship type has a checkbox with its numeric code and description. Check the types you want to analyze.

**Default:** All ship types are selected.

---

### Tab 3: Anomaly Types

**Purpose:** Select which anomalies to detect and configure detection thresholds.

#### Left Panel: Anomaly Type Selection

Checkboxes for each anomaly type:

- **AIS Beacon Off** - Detects when vessels turn off their AIS transponders
- **AIS Beacon On** - Detects when vessels suddenly reappear after being off
- **Speed Anomaly (Fast)** - Detects excessive travel distances (position jumps)
- **Speed Anomaly (Slow)** - Detects unusually slow travel (not currently implemented)
- **Course vs. Heading Inconsistency** - Detects large differences between COG and Heading
- **Loitering** - Detects vessels staying in a small area for extended periods
- **Rendezvous** - Detects two vessels meeting in close proximity
- **Identity Spoofing** - Detects multiple vessels using the same MMSI
- **Zone Violations** - Detects vessels entering restricted zones

**Buttons:**
- **Select All** - Enables all anomaly types
- **Deselect All** - Disables all anomaly types

#### Right Panel: Detection Thresholds

**Travel Distance Thresholds (nautical miles):**
- **Minimum** - Below this distance is considered "Slow" (default: 200.0 nm)
- **Maximum** - Above this distance is considered "Fast" (default: 550.0 nm)

**COG-Heading Inconsistency Thresholds:**
- **Maximum difference (degrees)** - Maximum allowed difference between COG and Heading (default: 45.0°)
- **Minimum speed for check (knots)** - Only check COG/Heading when vessel speed exceeds this (default: 10.0 knots)

**Note:** Other thresholds (loitering radius, rendezvous proximity, etc.) are configured in `config.ini`.

---

### Tab 4: Analysis Filters

**Purpose:** Narrow the analysis scope by geography, time, and specific vessels.

#### Geographic Boundaries

- **Latitude Range:**
  - **Min:** Minimum latitude (-90.0 to 90.0)
  - **Max:** Maximum latitude (-90.0 to 90.0)

- **Longitude Range:**
  - **Min:** Minimum longitude (-180.0 to 180.0)
  - **Max:** Maximum longitude (-180.0 to 180.0)

- **Draw Box on Map** Button:
  - Opens an interactive map in your browser
  - Draw a rectangle to set geographic bounds
  - Coordinates are automatically copied to clipboard
  - Paste into the Min/Max fields

**Default:** Full world coverage (-90 to 90 lat, -180 to 180 lon)

#### Time Filters

- **Hour of Day Range (0-24):**
  - **Start:** Beginning hour (default: 0)
  - **End:** Ending hour (default: 24)

Filters analysis to specific hours of the day.

#### Anomaly Filtering

- **Minimum Confidence Level (0-100):**
  - Only report anomalies above this confidence (default: 75)

- **Maximum Anomalies Per Vessel:**
  - Limit number of anomalies reported per vessel (default: 10000)

#### MMSI Filtering

- **Filter by MMSI:**
  - Enter comma-separated MMSI numbers (e.g., "123456789,987654321")
  - Leave empty to analyze all vessels

#### Use Defaults Button

Resets all filter values to defaults.

---

### Tab 5: Data

**Purpose:** Configure data sources and processing options.

#### File Locations

- **Data Directory:**
  - Local folder containing AIS data files
  - Click "Browse..." to select folder
  - Supports CSV and Parquet files

- **Output Directory:**
  - Where analysis results are saved
  - Click "Browse..." to select folder
  - Default: `C:\AIS_Data\Output` (Windows)

#### Data Source Selection

Radio buttons to choose data source:

- **Use Local Data Folder** - Read from local directory
- **Use AWS S3 Data Bucket** - Read from Amazon S3 (requires AWS credentials)

#### Amazon S3 Settings

**S3 Configuration:**
- **S3 URI:** Full S3 path (e.g., `s3://bucket-name/prefix/`)
- **Bucket Name:** S3 bucket name
- **Prefix:** Path prefix within bucket

**AWS Authentication (Access Keys):**
- **Access Key:** AWS access key ID
- **Secret Key:** AWS secret access key (hidden input)
- **Session Token:** Temporary session token (if using temporary credentials)

**Advanced Authentication Options:**
- **AWS Region:** AWS region name (default: us-east-1)

**Test S3 Connection Button:**
- Tests connectivity to S3 bucket
- Verifies credentials and permissions

#### Processing Options

- **Use Dask for distributed processing:**
  - Enables parallel processing for large files
  - Recommended for datasets > 1GB

- **Use GPU acceleration if available:**
  - Enables GPU acceleration when available
  - Shows GPU status (NVIDIA/AMD/None)
  - Provides "Install GPU Support" button if needed

**Note:** GPU support requires appropriate drivers and libraries (see Installation section).

---

### Tab 6: Zone Violations

**Purpose:** Manage restricted zones for zone violation detection.

#### Controls:

- **Add Zone** Button:
  - Opens dialog to create a new restricted zone
  - Enter zone name and coordinates
  - Option to draw zone on map

#### Zone List:

Each zone displays:
- **Checkbox:** "Selected" - Include this zone in analysis
- **Zone Name:** Descriptive name
- **Coordinates:** Latitude and longitude bounds
- **Edit Button:** Modify zone settings
- **Delete Button:** Remove zone (in edit dialog)

#### Zone Dialog:

When adding/editing a zone:

- **Zone Name:** Descriptive name (e.g., "Strait of Hormuz")
- **Latitude Range:**
  - **Min:** Minimum latitude
  - **Max:** Maximum latitude
- **Longitude Range:**
  - **Min:** Minimum longitude
  - **Max:** Maximum longitude
- **Draw Map** Button:
  - Opens interactive map to draw zone boundaries
  - Coordinates are copied to clipboard
- **Enter Zone Coordinates from Map** Button:
  - Opens map to manually enter coordinates
  - Right-click to paste coordinates

**Buttons:**
- **Select All** - Selects all zones
- **Deselect All** - Deselects all zones

**Note:** Only zones with "Selected" checked are included in analysis.

---

### Tab 7: Output Controls

**Purpose:** Configure what reports and visualizations are generated.

#### Report Outputs:

**Reports:**
- **Generate Statistics Excel** - Creates Excel file with statistics
- **Generate Statistics CSV** - Creates CSV file with statistics

**Maps:**
- **Generate Overall Map** - Creates HTML map of all anomalies
- **Generate Vessel Path Maps** - Creates individual maps for each vessel
- **Show Lat Long Grid** - Adds latitude/longitude grid to maps
- **Show Anomaly Heatmap** - Adds heatmap overlay to maps

**Charts:**
- **Generate Charts** - Enables chart generation
- **Generate Anomaly Type Chart** - Chart showing anomaly type distribution
- **Generate Vessel Anomaly Chart** - Chart showing top vessels with anomalies
- **Generate Date Anomaly Chart** - Chart showing anomalies over time

**Filtering:**
- **Filter To Anomaly Vessels Only** - Only include vessels with anomalies in reports

**Buttons:**
- **Select All** - Enables all outputs
- **Deselect All** - Disables all outputs

---

## Command-Line Interface (SFD.py)

For automated or scripted analysis, use the command-line interface:

```bash
python SFD.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD [options]
```

### Basic Usage

```bash
python SFD.py --start-date 2024-10-15 --end-date 2024-10-20
```

### Command-Line Options

#### Required Options:

- **--start-date** `YYYY-MM-DD` - Start date for analysis
- **--end-date** `YYYY-MM-DD` - End date for analysis

#### Optional Options:

**Configuration:**
- **--config** `path` - Path to config file (default: `config.ini`)

**Processing:**
- **--no-dask** - Disable Dask processing
- **--no-gpu** - Disable GPU processing
- **--force-gpu** - Force GPU usage (may cause errors if GPU unavailable)

**Debugging:**
- **--debug** - Enable debug logging
- **--show-warnings** - Show Python warnings

**Ship Types:**
- **--ship-types** `"70,80,81"` - Comma-separated ship type codes

**Output:**
- **--output-directory** `path` - Output directory for results

**Analysis Filters:**
- **--min-latitude** `float` - Minimum latitude
- **--max-latitude** `float` - Maximum latitude
- **--min-longitude** `float` - Minimum longitude
- **--max-longitude** `float` - Maximum longitude
- **--time-start-hour** `int` - Start hour (0-24)
- **--time-end-hour** `int` - End hour (0-24)
- **--min-confidence** `int` - Minimum confidence (0-100)
- **--max-anomalies-per-vessel** `int` - Max anomalies per vessel
- **--mmsi-list** `"123,456,789"` - Comma-separated MMSI list

**AWS S3:**
- **--access-key** `key` - AWS access key ID
- **--secret-key** `key` - AWS secret access key
- **--session-token** `token` - AWS session token
- **--region** `region` - AWS region (default: us-east-1)
- **--bucket** `name` - S3 bucket name
- **--prefix** `path` - S3 object prefix

### Example Commands

**Basic analysis:**
```bash
python SFD.py --start-date 2024-10-15 --end-date 2024-10-20
```

**With ship type filter:**
```bash
python SFD.py --start-date 2024-10-15 --end-date 2024-10-20 --ship-types "70,80"
```

**With geographic filter:**
```bash
python SFD.py --start-date 2024-10-15 --end-date 2024-10-20 --min-latitude 25.0 --max-latitude 27.0 --min-longitude 55.0 --max-longitude 57.5
```

**With S3 data source:**
```bash
python SFD.py --start-date 2024-10-15 --end-date 2024-10-20 --access-key YOUR_KEY --secret-key YOUR_SECRET --bucket my-bucket --prefix data/
```

**Debug mode:**
```bash
python SFD.py --start-date 2024-10-15 --end-date 2024-10-20 --debug
```

---

## Configuration File (config.ini)

The `config.ini` file stores all settings. It's automatically created on first run and can be edited manually or through the GUI.

### File Structure

```
[DEFAULT]
data_directory = C:/AIS_Data
output_directory = C:/AIS_Data/Output
start_date = 2024-10-15
end_date = 2024-10-17

[SHIP_FILTERS]
selected_ship_types = 70,80

[ANOMALY_THRESHOLDS]
min_travel_nm = 200.0
max_travel_nm = 550.0
cog_heading_max_diff = 45.0
min_speed_for_cog_check = 10.0
beacon_time_threshold_hours = 6.0

[ANOMALY_TYPES]
ais_beacon_off = False
ais_beacon_on = False
excessive_travel_distance_fast = False
cog-heading_inconsistency = False
loitering = False
rendezvous = False
identity_spoofing = False
zone_violations = True

[ANALYSIS_FILTERS]
min_latitude = -90.0
max_latitude = 90.0
min_longitude = -180.0
max_longitude = 180.0
time_start_hour = 0
time_end_hour = 24
min_confidence = 75
max_anomalies_per_vessel = 10
filter_mmsi_list = 

[OUTPUT_CONTROLS]
generate_statistics_excel = True
generate_statistics_csv = True
generate_overall_map = True
generate_vessel_path_maps = False
generate_charts = True
generate_anomaly_type_chart = True
generate_vessel_anomaly_chart = True
generate_date_anomaly_chart = True
filter_to_anomaly_vessels_only = False
show_lat_long_grid = False
show_anomaly_heatmap = True

[AWS]
use_s3 = True
s3_data_uri = s3://bucket-name/prefix/
s3_auth_method = keys
s3_access_key = 
s3_secret_key = 
s3_session_token = 
s3_bucket_name = bucket-name
s3_prefix = prefix/
s3_local_dir = output_directory
s3_region = us-east-1

[LOGGING]
log_level = INFO
log_file = SDF_GUI.log
suppress_warnings = True

[Processing]
use_gpu = True
use_dask = True

[ZONE_VIOLATIONS]
zone_0_name = Strait of Hormuz
zone_0_lat_min = 25.0
zone_0_lat_max = 27.0
zone_0_lon_min = 55.0
zone_0_lon_max = 57.5
zone_0_is_selected = False
```

### Section Descriptions

#### [DEFAULT]
- **data_directory** - Local data folder path
- **output_directory** - Results output folder path
- **start_date** - Default start date (YYYY-MM-DD)
- **end_date** - Default end date (YYYY-MM-DD)

#### [SHIP_FILTERS]
- **selected_ship_types** - Comma-separated ship type codes

#### [ANOMALY_THRESHOLDS]
- **min_travel_nm** - Minimum travel distance (nautical miles)
- **max_travel_nm** - Maximum travel distance (nautical miles)
- **cog_heading_max_diff** - Max COG/Heading difference (degrees)
- **min_speed_for_cog_check** - Minimum speed for COG check (knots)
- **beacon_time_threshold_hours** - Hours before beacon off/on is flagged

#### [ANOMALY_TYPES]
- **ais_beacon_off** - Enable/disable (True/False)
- **ais_beacon_on** - Enable/disable (True/False)
- **excessive_travel_distance_fast** - Enable/disable (True/False)
- **cog-heading_inconsistency** - Enable/disable (True/False)
- **loitering** - Enable/disable (True/False)
- **rendezvous** - Enable/disable (True/False)
- **identity_spoofing** - Enable/disable (True/False)
- **zone_violations** - Enable/disable (True/False)

#### [ANALYSIS_FILTERS]
- **min_latitude** - Minimum latitude (-90.0 to 90.0)
- **max_latitude** - Maximum latitude (-90.0 to 90.0)
- **min_longitude** - Minimum longitude (-180.0 to 180.0)
- **max_longitude** - Maximum longitude (-180.0 to 180.0)
- **time_start_hour** - Start hour (0-24)
- **time_end_hour** - End hour (0-24)
- **min_confidence** - Minimum confidence (0-100)
- **max_anomalies_per_vessel** - Max anomalies per vessel
- **filter_mmsi_list** - Comma-separated MMSI list (empty for all)

#### [OUTPUT_CONTROLS]
- **generate_statistics_excel** - Generate Excel report (True/False)
- **generate_statistics_csv** - Generate CSV report (True/False)
- **generate_overall_map** - Generate overall map (True/False)
- **generate_vessel_path_maps** - Generate vessel maps (True/False)
- **generate_charts** - Enable chart generation (True/False)
- **generate_anomaly_type_chart** - Anomaly type chart (True/False)
- **generate_vessel_anomaly_chart** - Vessel anomaly chart (True/False)
- **generate_date_anomaly_chart** - Date anomaly chart (True/False)
- **filter_to_anomaly_vessels_only** - Filter to anomalies only (True/False)
- **show_lat_long_grid** - Show grid on maps (True/False)
- **show_anomaly_heatmap** - Show heatmap on maps (True/False)

#### [AWS]
- **use_s3** - Use S3 data source (True/False)
- **s3_data_uri** - Full S3 URI
- **s3_auth_method** - Authentication method (keys)
- **s3_access_key** - AWS access key ID
- **s3_secret_key** - AWS secret access key
- **s3_session_token** - AWS session token (if using temporary credentials)
- **s3_bucket_name** - S3 bucket name
- **s3_prefix** - S3 object prefix
- **s3_local_dir** - Local cache directory
- **s3_region** - AWS region

#### [LOGGING]
- **log_level** - Logging level (DEBUG, INFO, WARNING, ERROR)
- **log_file** - Log file name
- **suppress_warnings** - Suppress Python warnings (True/False)

#### [Processing]
- **use_gpu** - Enable GPU acceleration (True/False)
- **use_dask** - Enable Dask processing (True/False)

#### [ZONE_VIOLATIONS]
- **zone_N_name** - Zone name (N = zone index)
- **zone_N_lat_min** - Minimum latitude
- **zone_N_lat_max** - Maximum latitude
- **zone_N_lon_min** - Minimum longitude
- **zone_N_lon_max** - Maximum longitude
- **zone_N_is_selected** - Include in analysis (True/False)

---

## Graphics and Icon Files

The application uses the following graphics files (all optional):

### Required for Full Functionality:

- **SFD_AI_banner.png**
  - **Location:** Same directory as SFD_GUI.py
  - **Usage:** Banner image displayed at top of GUI window
  - **Size:** Recommended 800x100 pixels or similar aspect ratio
  - **Format:** PNG with transparency support

- **SFDLoad.png**
  - **Location:** Same directory as SFD_GUI.py
  - **Usage:** Loading screen image shown during startup
  - **Size:** Recommended 400x300 pixels or similar
  - **Format:** PNG

- **SFD.ico**
  - **Location:** Same directory as SFD_GUI.py
  - **Usage:** Application icon for Windows taskbar/window
  - **Format:** ICO (Windows icon format)
  - **Note:** Only used on Windows systems

### Behavior:

- If graphics files are missing, the application will:
  - Display text labels instead of images
  - Continue to function normally
  - Log warnings (non-fatal)

### Creating Custom Graphics:

You can replace these files with your own graphics:
- Maintain similar dimensions for best appearance
- Use PNG format for images (supports transparency)
- Use ICO format for Windows icon
- Keep file names exactly as listed above

---

## Anomaly Types Explained

### 1. AIS Beacon Off

**Description:** Detects when a vessel's AIS transponder is turned off or stops transmitting.

**Detection Logic:**
- Vessel has position data, then suddenly stops transmitting
- Gap in transmission exceeds threshold (default: 6 hours)
- Vessel reappears later at a different location

**Use Case:** Identifying vessels that may be trying to avoid detection or engage in illicit activities.

**Threshold:** `beacon_time_threshold_hours` in config.ini (default: 6.0 hours)

---

### 2. AIS Beacon On

**Description:** Detects when a vessel suddenly reappears after being off.

**Detection Logic:**
- Vessel stops transmitting (beacon off detected)
- Vessel reappears at a different location
- Large distance traveled while "off"

**Use Case:** Identifying vessels that may have been operating without AIS or in restricted areas.

**Threshold:** `beacon_time_threshold_hours` in config.ini (default: 6.0 hours)

---

### 3. Speed Anomaly (Fast)

**Description:** Detects excessive travel distances between consecutive positions (position jumps).

**Detection Logic:**
- Calculates distance between consecutive AIS positions
- If distance exceeds maximum threshold, flags as anomaly
- Indicates possible data error, spoofing, or beacon manipulation

**Use Case:** Identifying vessels reporting impossible speeds or positions.

**Thresholds:**
- **Maximum:** `max_travel_nm` in config.ini (default: 550.0 nautical miles)
- **Minimum:** `min_travel_nm` in config.ini (default: 200.0 nautical miles)

---

### 4. Course vs. Heading Inconsistency

**Description:** Detects large differences between Course Over Ground (COG) and Heading.

**Detection Logic:**
- Compares reported COG (direction of movement) with Heading (direction vessel is pointing)
- Large differences may indicate:
  - Vessel drifting or being towed
  - Incorrect heading data
  - Potential spoofing

**Use Case:** Identifying vessels with suspicious navigation data.

**Thresholds:**
- **Maximum difference:** `cog_heading_max_diff` in config.ini (default: 45.0 degrees)
- **Minimum speed:** `min_speed_for_cog_check` in config.ini (default: 10.0 knots)

---

### 5. Loitering

**Description:** Detects vessels staying in a small area for extended periods.

**Detection Logic:**
- Tracks vessel positions over time
- Calculates if vessel remains within a radius for a duration
- May indicate:
  - Fishing operations
  - Waiting for rendezvous
  - Suspicious activity

**Use Case:** Identifying vessels engaged in suspicious loitering behavior.

**Thresholds:**
- **Radius:** `LOITERING_RADIUS_NM` in config.ini (default: 5.0 nautical miles)
- **Duration:** `LOITERING_DURATION_HOURS` in config.ini (default: 24.0 hours)

---

### 6. Rendezvous

**Description:** Detects two vessels meeting in close proximity.

**Detection Logic:**
- Tracks positions of all vessels
- Identifies when two vessels come within proximity threshold
- May indicate:
  - Cargo transfer
  - Refueling operations
  - Illicit transfers

**Use Case:** Identifying potential ship-to-ship transfers or meetings.

**Threshold:**
- **Proximity:** `RENDEZVOUS_PROXIMITY_NM` in config.ini (default: 0.5 nautical miles)

---

### 7. Identity Spoofing

**Description:** Detects multiple vessels using the same MMSI (Maritime Mobile Service Identity).

**Detection Logic:**
- Tracks MMSI numbers across all data
- Identifies when same MMSI appears in different locations simultaneously
- Indicates possible identity theft or spoofing

**Use Case:** Identifying vessels using stolen or duplicated identities.

**Threshold:** Automatic detection (no configurable threshold)

---

### 8. Zone Violations

**Description:** Detects vessels entering restricted zones.

**Detection Logic:**
- Checks vessel positions against defined restricted zones
- Flags when vessel enters a zone boundary
- Zones are user-defined in the Zone Violations tab

**Use Case:** Monitoring compliance with restricted areas (e.g., military zones, environmental protection areas).

**Configuration:** Defined in Zone Violations tab or `[ZONE_VIOLATIONS]` section of config.ini

---

## Output Files

After running an analysis, the following files may be generated in the output directory:

### Reports:

- **AIS_Anomalies_Summary.csv** - CSV file with all detected anomalies
- **AIS_Anomalies_Summary.xlsx** - Excel file with statistics and charts

### Maps:

- **All Anomalies Map.html** - Interactive HTML map showing all anomalies
- **Vessel Path Maps/** - Individual maps for each vessel (if enabled)
- **Anomally Heatmap** - Interactive anomaly heatmap of anomalies with muiltple comprehensive and daily view options

### Charts:

- **anomaly_types_distribution.png** - Bar chart showing anomaly type distribution
- **top_vessels_with_anomalies.png** - Chart showing vessels with most anomalies
- **anomalies_by_date.png** - Time series chart of anomalies over date range
- **anomalies_3d_bar_chart.png** - 3D bar chart (if plotly available)

### Logs:

- **sfd.log** - Analysis log file (from SFD.py)
- **SDF_GUI.log** - GUI log file (from SFD_GUI.py)

---

## Troubleshooting

### Common Issues:

#### 1. "Module not found" errors

**Solution:** Install missing dependencies:
```bash
pip install -r requirements.txt
```

#### 2. GUI won't start

**Solution:**
- Check Python version: `python --version` (needs 3.8+)
- On Linux, install tkinter: `sudo apt-get install python3-tk`
- Check for error messages in console

#### 3. GPU not detected

**Solution:**
- Verify GPU drivers are installed
- For NVIDIA: Install CUDA toolkit
- For AMD: Install HIP SDK
- Check GPU status in GUI: "Check for GPU Acceleration" button

#### 4. S3 connection fails

**Solution:**
- Verify AWS credentials are correct
- Check bucket name and prefix
- Verify IAM permissions for S3 access
- Test connection using "Test S3 Connection" button

#### 5. No anomalies detected

**Solution:**
- Check date range has available data
- Verify anomaly types are enabled
- Check ship type filters
- Review geographic filters (may be too restrictive)
- Check log files for errors

#### 6. Maps not generating

**Solution:**
- Verify folium is installed: `pip install folium`
- Check output directory permissions
- Review log files for errors

#### 7. Slow performance

**Solution:**
- Enable GPU acceleration (if available)
- Enable Dask processing for large files
- Reduce date range
- Filter by ship types or geography
- Reduce number of anomaly types

#### 8. Config file errors

**Solution:**
- Delete `config.ini` and let it regenerate
- Check for syntax errors (missing brackets, etc.)
- Verify all required sections exist
- Use GUI to save configuration

### Getting Help:

1. Check log files (`sfd.log`, `SDF_GUI.log`)
2. Enable debug mode: `--debug` flag or set `log_level = DEBUG` in config.ini
3. Review error messages in console
4. Verify all dependencies are installed correctly

---

## License and Credits

**Version:** 1.2 Beta  
**Team:** Dreadnaught  
**Author:** Chris Matherne  
**Event:** Datathon 2025

This software is provided as-is for the Datathon 2025 event.

---

## Additional Resources

- **AIS Data Format:** Consult AIS message format documentation
- **Maritime Regulations:** Refer to IMO and national maritime authority guidelines
- **GPU Acceleration:** See NVIDIA CUDA or AMD ROCm documentation
- **AWS S3:** See Amazon S3 documentation for bucket configuration

---

**End of README**


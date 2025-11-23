# AIS Shipping Fraud Detection System



## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Using SFD_GUI.py (Graphical Interface)](#using-sfd_guipy-graphical-interface)
5. [Using Advanced Analysis Tools](#using-advanced-analysis-tools)
6. [Command-Line Interface (SFD.py)](#command-line-interface-sfdpy)
7. [Configuration File (config.ini)](#configuration-file-configini)
8. [Anomaly Types Explained](#anomaly-types-explained)
9. [Output Files](#output-files)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The AIS Shipping Fraud Detection System analyzes Automatic Identification System (AIS) data to detect potentially fraudulent shipping activities. It identifies anomalies such as vessels turning off their AIS beacons (sudden disappearances), sudden reappearances, unusual travel distances, and large inconsistencies between reported Course Over Ground (COG) and Heading.

### Key Features

- **Interactive GUI**: User-friendly graphical interface for configuring and running analyses
- **Advanced Analysis Tools**: Post-analysis tools for deeper investigation of results
- **Multiple Anomaly Detection**: 8 different types of maritime anomalies
- **Flexible Data Sources**: Support for local files, NOAA downloads, and AWS S3
- **GPU Acceleration**: Optional GPU support for faster processing
- **Comprehensive Visualizations**: Interactive maps, charts, and reports
- **Data Caching**: Automatic caching system for faster subsequent analyses

### Data Availability

**Note:** In this test implementation, only data from October 2024 to March 2025 is available for use when using AWS.

---

## Installation

### Prerequisites

- Python 3.8 or higher (Python 3.14 fully supported)
- Windows, Linux, or macOS
- Internet connection (for NOAA data downloads)

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

Ensure the following files are in the same directory:

- **SFD_GUI.py** - Main graphical user interface
- **SFD.py** - Command-line analysis engine
- **advanced_analysis.py** - Advanced analysis tools module
- **config.ini** - Configuration file (created automatically if missing)

### Step 3: Optional Graphics Files

The following graphics files are optional but enhance the GUI appearance:

- **SFD_AI_banner.png** - Banner image for GUI
- **SFDLoad.png** - Loading screen image
- **SFD.ico** - Application icon (Windows only)

If these files are missing, the application will display text labels instead.

### Step 4: GPU Support (Optional)

For GPU acceleration:

**NVIDIA GPUs:**
- Install CUDA toolkit
- Packages will be installed via requirements.txt (cudf, cupy, cuml)

**AMD GPUs:**
- Install AMD HIP SDK
- Install cupy-rocm: `pip install cupy-rocm`
- Or install PyHIP: `pip install pyhip`

---

## Quick Start

1. **Launch the GUI:**
   ```bash
   python SFD_GUI.py
   ```

2. **Set your date range** on the Startup tab (e.g., 2024-10-21 to 2024-10-23)

3. **Configure your analysis:**
   - Select ship types on the Ship Types tab
   - Choose anomaly types on the Anomaly Types tab
   - Set data and output directories on the Data tab

4. **Run the analysis:**
   - Click the green "Run Analysis" button on the Startup tab
   - Wait for the analysis to complete

5. **View results:**
   - Results are saved to your output directory
   - After analysis completes, you can access Advanced Analysis tools

---

## Using SFD_GUI.py (Graphical Interface)

The GUI provides an intuitive interface for configuring and running analyses. It consists of 7 main tabs:

### Tab 1: Startup

**Purpose:** Set the analysis date range and execute the analysis.

#### Date Range Selection

- **Start Date** - Beginning of analysis period (format: YYYY-MM-DD)
- **End Date** - End of analysis period (format: YYYY-MM-DD)

If `tkcalendar` is installed, you'll see calendar widgets. Otherwise, use text entry fields.

#### Main Buttons

- **Run Analysis** (Green Button)
  - Executes the analysis with current settings
  - Shows progress in a popup window with download/extraction/conversion status
  - Results are saved to the output directory
  - After completion, you'll be asked if you want to conduct additional analysis

- **Save Configuration**
  - Saves all current settings to `config.ini`
  - Settings persist between sessions

- **Check for GPU Acceleration**
  - Tests GPU availability
  - Shows GPU status and installation options

- **Exit** (Red Button)
  - Closes the application
  - Prompts to delete temporary files if they exist

#### Description

The tab includes a description explaining the system's purpose and available data range.

---

### Tab 2: Ship Types

**Purpose:** Select which vessel types to include in the analysis.

#### Controls

- **Select All** - Selects all ship types
- **Deselect All** - Deselects all ship types
- **Category Buttons** - Select/deselect by category:
  - **WIG** (Wing in ground) - Types 20-29
  - **HSC** (High speed craft) - Types 40-49
  - **Special Purpose** - Types 50-59
  - **Passenger** - Types 60-69
  - **Cargo** - Types 70-79
  - **Tanker** - Types 80-89
  - **Other** - Types 90-99

Each ship type has a checkbox with its numeric code and description.

**Default:** All ship types are selected.

---

### Tab 3: Anomaly Types

**Purpose:** Select which anomalies to detect and configure detection thresholds.

#### Left Panel: Anomaly Type Selection

Checkboxes for each anomaly type:

- **AIS Beacon Off** - Detects when vessels turn off their AIS transponders
- **AIS Beacon On** - Detects when vessels suddenly reappear after being off
- **Excessive Travel Distance (Fast)** - Detects excessive travel distances (position jumps)
- **Excessive Travel Distance (Slow)** - Detects unusually slow travel
- **Course over Ground-Heading Inconsistency** - Detects large differences between COG and Heading
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
- **Maximum difference (degrees)** - Maximum allowed difference between COG and Heading (default: 45.0 degrees)
- **Minimum speed for check (knots)** - Only check COG/Heading when vessel speed exceeds this (default: 10.0 knots)

**AIS Beacon Threshold:**
- **Beacon time threshold (hours)** - Hours before beacon off/on is flagged (default: 6.0 hours)

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
- **Download from NOAA** - Automatically download AIS data from NOAA
- **Use AWS S3 Data Bucket** - Read from Amazon S3 (requires AWS credentials)

#### NOAA Data Download

When "Download from NOAA" is selected:
- System automatically downloads data for the specified date range
- Data is cached for future use
- Progress is shown in a download progress window
- Temporary files are stored in `AISDataTemp` directory

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

---

### Tab 6: Zone Violations

**Purpose:** Manage restricted zones for zone violation detection.

#### Controls

- **Add Zone** Button:
  - Opens dialog to create a new restricted zone
  - Supports both manual coordinate entry and drawing on an interactive map

#### Zone List

Each zone displays:
- **Checkbox:** "Selected" - Include this zone in analysis
- **Zone Name:** Descriptive name
- **Coordinates:** Latitude and longitude bounds
- **Edit Button:** Modify zone settings
- **Delete Button:** Remove zone (in edit dialog)

**Buttons:**
- **Select All** - Selects all zones
- **Deselect All** - Deselects all zones

**Note:** Only zones with "Selected" checked are included in analysis.

---

#### Creating a New Zone

You can create zones using two methods: **Manual Input** or **Draw on Map**. Both methods are accessed through the "Add Zone" button.

##### Method 1: Manual Input

**Step-by-Step Instructions:**

1. **Click "Add Zone" Button:**
   - Located on the Zone Violations tab
   - Opens the "Add Zone" dialog window

2. **Enter Zone Name:**
   - Type a descriptive name for the zone (e.g., "Strait of Hormuz", "Gulf of Mexico", "Mediterranean Sea")
   - Zone names must be unique

3. **Enter Coordinates Manually:**
   - **Latitude Min:** Enter the southernmost latitude (-90.0 to 90.0)
   - **Latitude Max:** Enter the northernmost latitude (must be greater than Min)
   - **Longitude Min:** Enter the westernmost longitude (-180.0 to 180.0)
   - **Longitude Max:** Enter the easternmost longitude (must be greater than Min)
   
   **Example:**
   - Strait of Hormuz:
     - Latitude Min: 25.0
     - Latitude Max: 27.0
     - Longitude Min: 55.0
     - Longitude Max: 57.5

4. **Set Zone Status:**
   - Check "Selected (use in analysis)" if you want this zone included in analysis
   - Uncheck to save the zone but exclude it from current analysis

5. **Save the Zone:**
   - Click "Save" button
   - The zone will be added to the zone list
   - Zone is automatically saved to `config.ini`

**Validation:**
- Zone name is required
- All coordinates must be valid numbers
- Latitude Min must be less than Latitude Max
- Longitude Min must be less than Longitude Max
- Zone names must be unique

---

##### Method 2: Draw on Map

**Step-by-Step Instructions:**

1. **Click "Add Zone" Button:**
   - Opens the "Add Zone" dialog window

2. **Enter Zone Name:**
   - Type a descriptive name for the zone
   - You can enter coordinates later

3. **Click "Draw Zone on Map" Button:**
   - A map will open in your default web browser
   - The map shows the world with drawing tools enabled

4. **Draw the Zone on the Map:**
   - **Locate the Drawing Toolbar:** Look for the drawing toolbar in the top-left corner of the map
   - **Select Rectangle Tool:** Click the rectangle icon in the toolbar
   - **Draw the Rectangle:**
     - Click and hold at one corner of the area you want to define
     - Drag to the opposite corner
     - Release to complete the rectangle
   - **View Coordinates:** After drawing, coordinates will appear in a box at the bottom-left of the map

5. **Copy Coordinates:**
   - Click the "Copy Coordinates" button in the coordinate display box
   - The coordinates will be copied to your clipboard in the format: `lat_min,lat_max,lon_min,lon_max`
   - The browser window may close automatically after copying

6. **Enter Coordinates in Dialog:**
   - A coordinate entry dialog will appear in the application
   - **Option A - Paste Coordinates:**
     - Paste the copied coordinates into the "Paste Coordinates" field
     - Click "Apply Coordinates" to populate all fields automatically
   - **Option B - Manual Entry:**
     - Manually enter the coordinates shown on the map into the individual fields:
       - Latitude Min
       - Latitude Max
       - Longitude Min
       - Longitude Max

7. **Set Zone Status:**
   - Check "Selected (use in analysis)" if you want this zone included in analysis

8. **Save the Zone:**
   - Click "Save" button in the main zone dialog
   - The zone will be added to the zone list
   - Zone is automatically saved to `config.ini`

**Tips for Drawing Zones:**
- Use the map zoom controls to get a closer view of the area
- You can pan the map by clicking and dragging
- The rectangle tool only allows rectangular zones (not polygons or circles)
- For irregular shapes, draw the smallest rectangle that encompasses the area
- Coordinates are displayed with 6 decimal places for precision

---

#### Editing an Existing Zone

**Step-by-Step Instructions:**

1. **Locate the Zone:**
   - Find the zone in the zone list on the Zone Violations tab

2. **Click "Edit" Button:**
   - Opens the "Edit Zone" dialog with current zone settings

3. **Modify Zone Settings:**
   - Change the zone name, coordinates, or selection status
   - You can use either manual input or "Draw Zone on Map" to update coordinates

4. **Save Changes:**
   - Click "Save" to update the zone
   - Changes are automatically saved to `config.ini`

5. **Delete Zone (Optional):**
   - While editing, click "Delete" button
   - Confirm deletion in the dialog
   - Zone will be removed from the list and `config.ini`

---

#### Zone Coordinate Format

Zones are defined as rectangular areas using:
- **Latitude Range:** From Min (south) to Max (north)
- **Longitude Range:** From Min (west) to Max (east)

**Coordinate Ranges:**
- Latitude: -90.0 (South Pole) to 90.0 (North Pole)
- Longitude: -180.0 to 180.0 (International Date Line)

**Example Zones:**
- **Strait of Hormuz:**
  - Lat: 25.0 to 27.0, Lon: 55.0 to 57.5
- **Gulf of Mexico:**
  - Lat: 18.0 to 30.0, Lon: -98.0 to -80.0
- **Mediterranean Sea:**
  - Lat: 30.0 to 46.0, Lon: -6.0 to 36.0

---

### Tab 7: Output Controls

**Purpose:** Configure what reports and visualizations are generated.

#### Report Outputs

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

## Using Advanced Analysis Tools

After completing an initial analysis, you can access Advanced Analysis tools to perform deeper investigations on your dataset. The Advanced Analysis interface is accessible from the results window after analysis completion, or can be launched directly.

### Accessing Advanced Analysis

**From Results Window:**
- After analysis completes, a dialog asks if you want to conduct additional analysis
- Click "Yes" to open the Advanced Analysis interface

**From Main GUI:**
- The Advanced Analysis tools are automatically available after your first analysis

### Advanced Analysis Interface Overview

The Advanced Analysis interface consists of 4 main tabs:

1. **Additional Outputs** - Generate additional reports and exports
2. **Further Analysis** - Perform statistical and pattern analysis
3. **Mapping Tools** - Create advanced map visualizations
4. **Vessel-Specific Analysis** - Analyze individual vessels in detail

---

### Tab 1: Additional Outputs

This tab provides tools to generate additional reports and data exports from your analysis results.

#### Export Full Dataset to CSV

**Purpose:** Export the complete analysis dataset to CSV format.

**How to Use:**
1. Click the "Export Full Dataset to CSV" button
2. A file dialog will appear to choose the save location
3. The export process will show progress
4. Large datasets are exported in chunks to handle memory efficiently

**Output:** A CSV file containing all data from the analysis.

#### Generate Summary Report

**Purpose:** Create a comprehensive summary report with key findings and statistics.

**How to Use:**
1. Click the "Generate Summary Report" button
2. The system will analyze the dataset and generate a report
3. Progress is shown in a dialog window
4. The report is saved as an HTML file

**Output:** An HTML report file with:
- Total anomalies detected
- Anomaly type breakdown
- Top vessels with anomalies
- Geographic distribution
- Temporal patterns

#### Export Vessel Statistics

**Purpose:** Export vessel-specific statistics to Excel format.

**How to Use:**
1. Click the "Export Vessel Statistics" button
2. The system will process all vessels in the dataset
3. Statistics are compiled and exported
4. Progress is shown during processing

**Output:** An Excel file (or CSV if openpyxl is not available) containing:
- Vessel MMSI and names
- Total anomalies per vessel
- Anomaly type breakdown per vessel
- Average speeds and distances
- Time ranges

#### Generate Anomaly Timeline

**Purpose:** Create a timeline visualization of anomalies.

**How to Use:**
1. Click the "Generate Anomaly Timeline" button
2. The system will create an interactive timeline
3. Progress is shown during generation

**Output:** An HTML file with an interactive timeline showing:
- Anomalies over time
- Anomaly types by time period
- Vessel activity patterns

---

### Tab 2: Further Analysis

This tab provides advanced statistical and pattern analysis tools.

#### Anomaly Correlation Analysis

**Purpose:** Analyze correlations between vessel types and anomaly types.

**How to Use:**
1. Click the "Anomaly Correlation Analysis" button
2. A dialog will open with selection options:
   - **Vessel Types:** Select up to 2 vessel types (only types present in your data are available)
   - **Anomaly Types:** Select up to 2 anomaly types (only types detected in your analysis are available)
3. Click "Analyze" to generate the correlation analysis
4. Results are displayed in an interactive HTML chart

**Output:** An HTML file with correlation visualizations showing relationships between selected vessel and anomaly types.

**Note:** Options are automatically filtered to only show vessel types and anomaly types that exist in your dataset.

#### Temporal Pattern Analysis

**Purpose:** Analyze patterns over time, including hourly and daily distributions.

**How to Use:**
1. Click the "Temporal Pattern Analysis" button
2. The system will analyze temporal patterns in your data
3. Progress is shown during analysis

**Output:** An HTML file with interactive charts showing:
- Anomalies by hour of day
- Anomalies by day of week
- Trends over time
- Peak activity periods

#### Vessel Behavior Clustering

**Purpose:** Apply clustering algorithms to identify similar vessel behaviors.

**How to Use:**
1. Click the "Vessel Behavior Clustering" button
2. A dialog will open:
   - **Vessel Types:** Select vessel types to include (all types 20-99 are shown, with indicators for which are in your data)
   - **Number of Clusters:** Enter the desired number of clusters (default: 5)
3. Click "Perform Clustering" to run the analysis
4. Results are displayed in an interactive visualization

**Output:** An HTML file (or CSV if plotly is not available) showing:
- Vessel clusters based on behavior patterns
- Cluster characteristics
- Vessel assignments to clusters

**Note:** Requires scikit-learn to be installed. If not available, the feature will be disabled.

#### Anomaly Frequency Analysis

**Purpose:** Analyze frequency and distribution of different anomaly types.

**How to Use:**
1. Click the "Anomaly Frequency Analysis" button
2. The system will analyze anomaly frequencies
3. Progress is shown during analysis

**Output:** An HTML file with charts showing:
- Frequency of each anomaly type
- Distribution patterns
- Statistical summaries

#### Create Custom Chart

**Purpose:** Create custom charts with various types and data selections.

**How to Use:**
1. Click the "Create Custom Chart" button
2. A dialog will open with options:
   - **Chart Type:** Select from bar, scatter, line, pie, stacked_bar, timeline, histogram, or box
   - **X Column:** Select the column for x-axis
   - **Y Column:** Select the column for y-axis (if applicable)
   - **Color Column:** Select column for color grouping (optional)
   - **Group By:** Select column to group by (optional)
   - **Aggregation:** Select aggregation function (count, sum, mean, max, min)
   - **Title:** Enter chart title
3. Click "Create Chart" to generate
4. The chart is saved as an HTML file

**Output:** An interactive HTML chart based on your specifications.

**Available Chart Types:**
- **Bar:** Bar chart for categorical data
- **Scatter:** Scatter plot for relationships
- **Line:** Line chart for trends
- **Pie:** Pie chart for proportions
- **Stacked Bar:** Stacked bar chart for comparisons
- **Timeline:** Timeline visualization
- **Histogram:** Distribution histogram
- **Box:** Box plot for distributions

---

### Tab 3: Mapping Tools

This tab provides advanced mapping and visualization tools.

#### Full Spectrum Anomaly Map

**Purpose:** Create a comprehensive map showing all anomalies with various visualization options.

**How to Use:**
1. Click the "Full Spectrum Anomaly Map" button
2. A dialog will open with options:
   - **Show Anomaly Pins:** Check to show individual anomaly markers
   - **Show Heatmap Overlay:** Check to show heatmap visualization
3. Click "Create Map" to generate
4. The map opens in your default web browser

**Output:** An interactive HTML map with:
- All anomalies from your analysis
- Optional pin markers for each anomaly
- Optional heatmap overlay showing anomaly density
- Clickable markers with anomaly details

#### Vessel-Specific Maps

**Purpose:** Create maps focused on specific vessels by MMSI.

**How to Use:**
1. Click the "Create Vessel Map" button
2. A dialog will open:
   - **MMSI:** Enter the vessel's MMSI number
   - **Map Type:** Select from:
     - **Path Map:** Shows vessel's path over time
     - **Anomaly Map:** Shows anomalies for this vessel
     - **Heatmap:** Shows heatmap of vessel activity
3. Click "Create Map" to generate

**Output:** An interactive HTML map showing the selected vessel's data.

**Top Vessels List:**
- Below the vessel map button, you'll see lists of top 10 vessels by anomaly type
- Click "Copy" next to any MMSI to copy it to clipboard
- Paste the MMSI into the vessel map dialog

#### Filtered Maps

**Purpose:** Create path/anomaly/heat-maps for specific anomaly types and/or vessel types.

**How to Use:**
1. Click the "Create Filtered Map" button
2. A dialog will open:
   - **Anomaly Types:** Select one or more anomaly types (only types from your analysis are available)
   - **Vessel Types:** Select one or more vessel types (only types from your analysis are available)
   - **Map Type:** Select Path Map, Anomaly Map, or Heatmap
   - **Specific Vessel (Optional):** Enter MMSI to filter to a specific vessel
3. Click "Create Map" to generate

**Output:** An interactive HTML map filtered by your selections.

**Note:** Only anomaly types and vessel types that were included in your original analysis are available for selection.

---

### Tab 4: Vessel-Specific Analysis

This tab provides tools for detailed analysis of individual vessels.

#### Extended Time Range Analysis

**Purpose:** Analyze additional days of data for a specific vessel beyond the current date range.

**How to Use:**
1. Enter the vessel's **MMSI** number
2. Select **Additional Days** using the calendar pickers:
   - **Start Date:** Beginning of extended analysis period
   - **End Date:** End of extended analysis period
3. Click "Analyze" to run the extended analysis
4. Progress is shown during analysis

**Output:** Analysis results for the extended time period, including:
- Additional anomalies detected
- Extended path visualization
- Updated statistics

**Note:** The extended analysis uses cached data if available, or downloads additional data if needed.

#### AI Predicted Path (Placeholder)

**Purpose:** Use AI to predict the next 48-hour path for a specific vessel.

**Status:** This feature is currently a placeholder for future development.

**How to Use:**
1. Enter the vessel's **MMSI** number
2. Click "Predict Path" (when implemented)

**Output:** (To be implemented) Predicted vessel path visualization.

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

**Advanced Analysis:**
- **--advanced-analysis** `feature` - Run advanced analysis feature:
  - `export-full-dataset` - Export full dataset to CSV
  - `summary-report` - Generate summary report
  - `vessel-statistics` - Export vessel statistics
  - `anomaly-timeline` - Generate anomaly timeline
  - `temporal-patterns` - Temporal pattern analysis
  - `vessel-clustering` - Vessel behavior clustering
  - `anomaly-frequency` - Anomaly frequency analysis
  - `full-spectrum-map` - Create full spectrum map
  - `vessel-map` - Create vessel-specific map
- **--vessel-mmsi** `int` - MMSI for vessel-specific analysis
- **--map-type** `type` - Map type (path, anomaly, heatmap)
- **--no-show-pins** - Hide pins on full spectrum map
- **--no-show-heatmap** - Hide heatmap on full spectrum map
- **--extended-start-date** `YYYY-MM-DD` - Start date for extended analysis
- **--extended-end-date** `YYYY-MM-DD` - End date for extended analysis
- **--n-clusters** `int` - Number of clusters for clustering (default: 5)

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

**Advanced analysis - generate summary report:**
```bash
python SFD.py --advanced-analysis summary-report
```

**Advanced analysis - create vessel map:**
```bash
python SFD.py --advanced-analysis vessel-map --vessel-mmsi 366215000 --map-type path
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
ais_beacon_off = True
ais_beacon_on = True
excessive_travel_distance_fast = True
excessive_travel_distance_slow = True
cog-heading_inconsistency = True
loitering = False
rendezvous = False
identity_spoofing = False
zone_violations = False

[ANALYSIS_FILTERS]
min_latitude = -90.0
max_latitude = 90.0
min_longitude = -180.0
max_longitude = 180.0
time_start_hour = 0
time_end_hour = 24
min_confidence = 75
max_anomalies_per_vessel = 10000
filter_mmsi_list = 

[OUTPUT_CONTROLS]
generate_statistics_excel = True
generate_statistics_csv = False
generate_overall_map = True
generate_vessel_path_maps = False
generate_charts = False
generate_anomaly_type_chart = False
generate_vessel_anomaly_chart = False
generate_date_anomaly_chart = False
filter_to_anomaly_vessels_only = False
show_lat_long_grid = False
show_anomaly_heatmap = True

[AWS]
use_s3 = False
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

See the old SFDREADME.md for detailed descriptions of each configuration section. The structure remains the same, but some deprecated options have been removed.

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

---

### 4. Speed Anomaly (Slow)

**Description:** Detects unusually slow travel between positions.

**Detection Logic:**
- Calculates distance between consecutive AIS positions
- If distance is below minimum threshold, flags as anomaly
- May indicate stationary vessel or data issues

**Use Case:** Identifying vessels with suspiciously slow movement patterns.

**Thresholds:**
- **Minimum:** `min_travel_nm` in config.ini (default: 200.0 nautical miles)

---

### 5. Course vs. Heading Inconsistency

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

### 6. Loitering

**Description:** Detects vessels staying in a small area for extended periods.

**Detection Logic:**
- Tracks vessel positions over time
- Calculates if vessel remains within a radius for a duration
- May indicate:
  - Fishing operations
  - Waiting for rendezvous
  - Suspicious activity

**Use Case:** Identifying vessels engaged in suspicious loitering behavior.

---

### 7. Rendezvous

**Description:** Detects two vessels meeting in close proximity.

**Detection Logic:**
- Tracks positions of all vessels
- Identifies when two vessels come within proximity threshold
- May indicate:
  - Cargo transfer
  - Refueling operations
  - Illicit transfers

**Use Case:** Identifying potential ship-to-ship transfers or meetings.

---

### 8. Identity Spoofing

**Description:** Detects multiple vessels using the same MMSI (Maritime Mobile Service Identity).

**Detection Logic:**
- Tracks MMSI numbers across all data
- Identifies when same MMSI appears in different locations simultaneously
- Indicates possible identity theft or spoofing

**Use Case:** Identifying vessels using stolen or duplicated identities.

---

### 9. Zone Violations

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
- **AIS_Anomalies_Summary.xlsx** - Excel file with statistics and charts (if enabled)

### Maps:

- **All Anomalies Map.html** - Interactive HTML map showing all anomalies
- **Vessel Path Maps/** - Individual maps for each vessel (if enabled)
- **Anomaly Heatmap.html** - Interactive anomaly heatmap with multiple comprehensive and daily view options

### Charts:

- **anomaly_types_distribution.png** - Bar chart showing anomaly type distribution (if enabled)
- **top_vessels_with_anomalies.png** - Chart showing vessels with most anomalies (if enabled)
- **anomalies_by_date.png** - Time series chart of anomalies over date range (if enabled)

### Logs:

- **sfd.log** - Analysis log file (from SFD.py)
- **ais_fraud_detection_gui.log** - GUI log file (from SFD_GUI.py)
- **advanced_analysis.log** - Advanced analysis log file (from advanced_analysis.py)

### Temporary Files:

- **AISDataTemp/** - Temporary directory for downloads and processing
  - Automatically cleaned up after processing
  - User is prompted to delete on application exit

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

#### 9. Advanced Analysis not available

**Solution:**
- Ensure `advanced_analysis.py` is in the same directory as `SFD_GUI.py`
- Check that all dependencies are installed
- Review `advanced_analysis.log` for errors
- Verify that an initial analysis has been completed

#### 10. Download progress stuck

**Solution:**
- Check internet connection
- Verify NOAA website is accessible
- Check `AISDataTemp` directory for partial downloads
- Review log files for download errors
- Try clearing cache and re-downloading

### Getting Help:

1. Check log files (`sfd.log`, `ais_fraud_detection_gui.log`, `advanced_analysis.log`)
2. Enable debug mode: `--debug` flag or set `log_level = DEBUG` in config.ini
3. Review error messages in console
4. Verify all dependencies are installed correctly

---

## License and Credits

**Version:** 2.0 MVP  
**Team:** Dreadnaught  
**Authors:** Chris Matherne, Zhoang, Alex 
**Event:** Datathon 2025

This software is provided as-is for the Datathon 2025 event.

---

## Additional Resources

- **AIS Data Format:** Consult AIS message format documentation
- **Maritime Regulations:** Refer to IMO and national maritime authority guidelines
- **GPU Acceleration:** See NVIDIA CUDA or AMD ROCm documentation
- **AWS S3:** See Amazon S3 documentation for bucket configuration
- **NOAA AIS Data:** See NOAA AIS data documentation for data format and availability

---

**End of README**


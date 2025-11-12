"""
Major Bodies of Water - Geographic Definitions

Defines major oceans, seas, and water bodies as simplified lat/lon polygons
for efficient geographic queries and vessel location identification.

Each polygon is defined as a list of [longitude, latitude] coordinate pairs.
Polygons are simplified to use fewer points while maintaining reasonable accuracy.

Coordinate format: [longitude, latitude] (standard GeoJSON format)
- Longitude: -180 to 180 (negative = West, positive = East)
- Latitude: -90 to 90 (negative = South, positive = North)
"""

# ============================================================
# MAJOR OCEANS
# ============================================================

PACIFIC_OCEAN = {
    "name": "Pacific Ocean",
    "type": "ocean",
    "polygon": [
        [-180, 65], [-180, -60], [-70, -60], [-70, -15],  # East Pacific
        [-80, 0], [-82, 10], [-90, 15], [-110, 20],  # Central America
        [-120, 25], [-125, 30], [-130, 35], [-140, 40],  # West Coast USA
        [-145, 45], [-150, 50], [-160, 55], [-170, 60],  # Alaska
        [-180, 65], [180, 65],  # Bering Strait
        [180, 60], [170, 55], [165, 50], [160, 45],  # East Russia
        [155, 40], [150, 35], [145, 30], [140, 25],  # Japan
        [135, 20], [130, 15], [125, 10], [120, 5],  # Philippines
        [115, 0], [115, -10], [120, -20], [125, -30],  # Indonesia/Australia
        [130, -40], [140, -45], [150, -50], [160, -55],  # South Pacific
        [170, -60], [180, -60], [-180, -60]  # Close polygon
    ]
}

ATLANTIC_OCEAN = {
    "name": "Atlantic Ocean",
    "type": "ocean",
    "polygon": [
        [-100, 70], [-90, 72], [-80, 73], [-70, 72],  # North America/Greenland
        [-60, 70], [-50, 75], [-40, 75], [-30, 70],  # Greenland/Iceland
        [-20, 65], [-10, 60], [0, 58], [10, 55],  # North Atlantic/Europe
        [15, 50], [10, 45], [5, 40], [0, 35],  # Western Europe
        [-5, 30], [-10, 25], [-15, 20], [-18, 15],  # NW Africa
        [-20, 10], [-22, 5], [-25, 0], [-28, -5],  # W Africa
        [-30, -10], [-32, -20], [-35, -30], [-38, -40],  # SW Atlantic
        [-45, -50], [-50, -55], [-60, -60], [-70, -60],  # South Atlantic
        [-70, -50], [-72, -40], [-74, -30], [-75, -20],  # South America
        [-76, -10], [-77, 0], [-78, 10], [-80, 20],  # South America
        [-82, 30], [-85, 40], [-88, 50], [-92, 60],  # North America
        [-96, 65], [-100, 70]  # Close polygon
    ]
}

INDIAN_OCEAN = {
    "name": "Indian Ocean",
    "type": "ocean",
    "polygon": [
        [20, 30], [30, 25], [35, 20], [40, 15],  # Red Sea entry
        [45, 12], [50, 10], [55, 10], [60, 8],  # Arabian Sea
        [65, 8], [70, 8], [75, 8], [80, 10],  # India
        [85, 10], [90, 12], [95, 10], [100, 5],  # Bay of Bengal
        [105, 0], [110, -5], [115, -10], [120, -20],  # Indonesia
        [125, -30], [130, -40], [120, -45], [110, -48],  # South Indian
        [100, -50], [90, -50], [80, -48], [70, -45],  # South Indian
        [60, -42], [50, -38], [40, -35], [30, -35],  # SW Indian
        [25, -32], [20, -30], [18, -25], [17, -20],  # E Africa
        [16, -10], [15, 0], [15, 10], [18, 20],  # E Africa
        [20, 25], [20, 30]  # Close polygon
    ]
}

ARCTIC_OCEAN = {
    "name": "Arctic Ocean",
    "type": "ocean",
    "polygon": [
        [-180, 65], [-180, 90], [180, 90], [180, 65],  # Arctic Circle
        [170, 65], [160, 63], [150, 62], [140, 62],  # Russia
        [130, 63], [120, 65], [110, 67], [100, 68],  # North Russia
        [90, 70], [80, 72], [70, 73], [60, 74],  # Barents Sea
        [50, 75], [40, 75], [30, 75], [20, 75],  # Svalbard
        [10, 75], [0, 75], [-10, 74], [-20, 73],  # Greenland Sea
        [-30, 72], [-40, 75], [-50, 77], [-60, 78],  # NW Greenland
        [-70, 78], [-80, 77], [-90, 76], [-100, 75],  # Canadian Arctic
        [-110, 74], [-120, 73], [-130, 71], [-140, 69],  # Alaska
        [-150, 67], [-160, 66], [-170, 65], [-180, 65]  # Bering Strait
    ]
}

SOUTHERN_OCEAN = {
    "name": "Southern Ocean",
    "type": "ocean",
    "polygon": [
        [-180, -60], [-180, -90], [180, -90], [180, -60],  # Antarctic Circle
        [170, -60], [160, -58], [150, -57], [140, -56],  # East Antarctic
        [130, -56], [120, -57], [110, -58], [100, -59],  # Indian sector
        [90, -60], [80, -60], [70, -60], [60, -60],  # Indian sector
        [50, -60], [40, -60], [30, -60], [20, -60],  # Atlantic sector
        [10, -60], [0, -60], [-10, -60], [-20, -60],  # Atlantic sector
        [-30, -60], [-40, -60], [-50, -60], [-60, -60],  # Atlantic sector
        [-70, -60], [-80, -60], [-90, -60], [-100, -60],  # Pacific sector
        [-110, -60], [-120, -60], [-130, -60], [-140, -60],  # Pacific sector
        [-150, -60], [-160, -60], [-170, -60], [-180, -60]  # Pacific sector
    ]
}

# ============================================================
# MAJOR SEAS - ATLANTIC
# ============================================================

CARIBBEAN_SEA = {
    "name": "Caribbean Sea",
    "type": "sea",
    "parent": "Atlantic Ocean",
    "polygon": [
        [-87, 22], [-82, 23], [-78, 22], [-74, 21],  # Cuba
        [-70, 20], [-68, 19], [-66, 18], [-64, 17],  # Lesser Antilles
        [-62, 15], [-61, 13], [-60, 11], [-60, 9],  # Venezuela
        [-62, 9], [-65, 9], [-68, 9], [-71, 9],  # Colombia
        [-75, 10], [-78, 11], [-81, 13], [-84, 15],  # Central America
        [-86, 17], [-87, 19], [-87, 22]  # Mexico
    ]
}

GULF_OF_MEXICO = {
    "name": "Gulf of Mexico",
    "type": "gulf",
    "parent": "Atlantic Ocean",
    "polygon": [
        [-98, 30], [-97, 28], [-96, 27], [-95, 26],  # Texas
        [-94, 25], [-92, 24], [-90, 24], [-88, 25],  # Louisiana
        [-86, 26], [-84, 27], [-82, 28], [-81, 26],  # Florida
        [-82, 24], [-84, 22], [-87, 21], [-90, 20],  # Yucatan
        [-93, 19], [-96, 19], [-98, 20], [-98, 22],  # Mexico
        [-98, 25], [-98, 28], [-98, 30]  # Close
    ]
}

MEDITERRANEAN_SEA = {
    "name": "Mediterranean Sea",
    "type": "sea",
    "parent": "Atlantic Ocean",
    "polygon": [
        [-6, 36], [-4, 36], [-2, 37], [0, 38],  # Gibraltar to Spain
        [3, 39], [6, 40], [9, 41], [12, 42],  # France/Italy
        [15, 41], [18, 40], [21, 39], [24, 38],  # Adriatic/Greece
        [27, 36], [30, 35], [33, 34], [36, 33],  # E Mediterranean
        [36, 31], [34, 30], [32, 30], [30, 31],  # Israel/Egypt
        [27, 32], [24, 33], [21, 34], [18, 35],  # Libya
        [15, 35], [12, 35], [9, 36], [6, 36],  # Tunisia/Algeria
        [3, 36], [0, 36], [-3, 36], [-6, 36]  # Morocco
    ]
}

BLACK_SEA = {
    "name": "Black Sea",
    "type": "sea",
    "parent": "Mediterranean Sea",
    "polygon": [
        [27, 42], [29, 43], [31, 44], [33, 45],  # Turkey
        [36, 46], [38, 46], [40, 45], [41, 44],  # Georgia
        [41, 43], [40, 42], [38, 41], [36, 41],  # Turkey
        [34, 42], [32, 42], [30, 42], [28, 42], [27, 42]  # Bulgaria/Romania
    ]
}

NORTH_SEA = {
    "name": "North Sea",
    "type": "sea",
    "parent": "Atlantic Ocean",
    "polygon": [
        [-4, 50], [-2, 51], [0, 52], [2, 53],  # English Channel
        [4, 54], [5, 55], [6, 56], [7, 57],  # Netherlands/Germany
        [8, 58], [9, 58], [10, 58], [9, 59],  # Denmark
        [8, 60], [6, 61], [4, 61], [2, 60],  # Norway
        [0, 59], [-2, 58], [-3, 56], [-4, 54], [-4, 52], [-4, 50]  # Scotland
    ]
}

BALTIC_SEA = {
    "name": "Baltic Sea",
    "type": "sea",
    "parent": "Atlantic Ocean",
    "polygon": [
        [10, 54], [11, 55], [12, 56], [14, 57],  # Germany/Denmark
        [16, 58], [18, 59], [20, 60], [22, 61],  # Sweden
        [24, 62], [26, 63], [27, 64], [28, 65],  # Finland
        [28, 64], [27, 63], [26, 62], [24, 61],  # Gulf of Bothnia
        [22, 60], [20, 59], [18, 58], [16, 57],  # Estonia
        [14, 56], [12, 55], [10, 54]  # Poland
    ]
}

# ============================================================
# MAJOR SEAS - PACIFIC
# ============================================================

SOUTH_CHINA_SEA = {
    "name": "South China Sea",
    "type": "sea",
    "parent": "Pacific Ocean",
    "polygon": [
        [99, 22], [102, 21], [105, 20], [108, 19],  # Vietnam
        [110, 18], [112, 16], [114, 14], [116, 12],  # Philippines
        [118, 10], [119, 8], [120, 6], [120, 4],  # Borneo
        [118, 2], [116, 1], [114, 1], [112, 1],  # Indonesia
        [110, 2], [108, 3], [106, 4], [104, 5],  # Malaysia
        [102, 7], [100, 9], [99, 11], [99, 14],  # Thailand
        [99, 17], [99, 20], [99, 22]  # Close
    ]
}

EAST_CHINA_SEA = {
    "name": "East China Sea",
    "type": "sea",
    "parent": "Pacific Ocean",
    "polygon": [
        [120, 26], [121, 27], [122, 28], [123, 29],  # Taiwan
        [124, 30], [125, 31], [126, 32], [127, 33],  # Korea
        [128, 34], [129, 35], [130, 35], [129, 34],  # Japan
        [128, 33], [127, 32], [126, 31], [125, 30],  # Okinawa
        [124, 29], [123, 28], [122, 27], [121, 26], [120, 26]  # Close
    ]
}

SEA_OF_JAPAN = {
    "name": "Sea of Japan (East Sea)",
    "type": "sea",
    "parent": "Pacific Ocean",
    "polygon": [
        [128, 35], [129, 36], [130, 37], [131, 38],  # Korea
        [132, 39], [133, 40], [134, 41], [136, 42],  # Japan
        [138, 43], [140, 44], [141, 45], [141, 44],  # Hokkaido
        [140, 43], [139, 42], [138, 41], [137, 40],  # Sea center
        [136, 39], [135, 38], [133, 37], [131, 36],  # Close
        [129, 35], [128, 35]
    ]
}

BERING_SEA = {
    "name": "Bering Sea",
    "type": "sea",
    "parent": "Pacific Ocean",
    "polygon": [
        [-180, 55], [-178, 56], [-175, 57], [-172, 58],  # Aleutians
        [-170, 59], [-168, 60], [-166, 61], [-164, 62],  # Alaska
        [-162, 63], [-160, 64], [-158, 65], [-160, 66],  # Bering Strait
        [-165, 66], [-170, 65], [-175, 64], [180, 63],  # Russia
        [178, 62], [176, 61], [174, 60], [172, 59],  # Kamchatka
        [170, 58], [168, 57], [165, 56], [162, 55],  # Close
        [-180, 55]
    ]
}

# ============================================================
# MAJOR SEAS - INDIAN OCEAN
# ============================================================

ARABIAN_SEA = {
    "name": "Arabian Sea",
    "type": "sea",
    "parent": "Indian Ocean",
    "polygon": [
        [50, 25], [52, 24], [55, 23], [58, 22],  # Arabian Peninsula
        [60, 21], [62, 20], [64, 19], [66, 18],  # Oman
        [68, 17], [70, 16], [72, 15], [73, 13],  # India
        [73, 11], [72, 9], [70, 8], [68, 7],  # Maldives
        [66, 8], [64, 9], [62, 10], [60, 11],  # Center
        [58, 12], [56, 14], [54, 16], [52, 18],  # Yemen
        [50, 20], [50, 22], [50, 25]  # Close
    ]
}

BAY_OF_BENGAL = {
    "name": "Bay of Bengal",
    "type": "bay",
    "parent": "Indian Ocean",
    "polygon": [
        [80, 22], [82, 21], [84, 20], [86, 19],  # India
        [88, 18], [90, 17], [92, 16], [94, 15],  # Myanmar
        [96, 13], [97, 11], [98, 9], [98, 7],  # Andaman Sea
        [96, 6], [94, 6], [92, 7], [90, 8],  # Sumatra
        [88, 9], [86, 11], [84, 13], [82, 15],  # Sri Lanka
        [80, 17], [80, 19], [80, 22]  # Close
    ]
}

PERSIAN_GULF = {
    "name": "Persian Gulf (Arabian Gulf)",
    "type": "gulf",
    "parent": "Indian Ocean",
    "polygon": [
        [48, 30], [49, 29], [50, 29], [51, 28],  # Kuwait
        [52, 27], [53, 27], [54, 26], [55, 26],  # UAE
        [56, 26], [56, 25], [56, 24], [55, 24],  # Oman
        [54, 25], [53, 25], [52, 26], [51, 27],  # Qatar
        [50, 28], [49, 29], [48, 30]  # Iran/Iraq
    ]
}

RED_SEA = {
    "name": "Red Sea",
    "type": "sea",
    "parent": "Indian Ocean",
    "polygon": [
        [32, 30], [33, 28], [34, 26], [35, 24],  # Egypt
        [36, 22], [37, 20], [38, 18], [39, 16],  # Sudan
        [40, 14], [41, 13], [42, 13], [43, 13],  # Eritrea/Yemen
        [43, 14], [43, 15], [42, 17], [41, 19],  # Center
        [40, 21], [39, 23], [38, 25], [37, 27],  # Saudi Arabia
        [36, 28], [35, 29], [33, 30], [32, 30]  # Sinai
    ]
}

# ============================================================
# STRAITS AND PASSAGES
# ============================================================

STRAIT_OF_GIBRALTAR = {
    "name": "Strait of Gibraltar",
    "type": "strait",
    "polygon": [
        [-6, 36], [-5.5, 36], [-5.5, 35.8], [-6, 35.8], [-6, 36]
    ]
}

STRAIT_OF_MALACCA = {
    "name": "Strait of Malacca",
    "type": "strait",
    "polygon": [
        [98, 7], [99, 6], [100, 5], [101, 4], [102, 3],
        [103, 2], [104, 1.5], [103.5, 1], [102.5, 1.5],
        [101.5, 2], [100.5, 3], [99.5, 4], [98.5, 5], [98, 6], [98, 7]
    ]
}

STRAIT_OF_HORMUZ = {
    "name": "Strait of Hormuz",
    "type": "strait",
    "polygon": [
        [56, 26.5], [57, 26.5], [57, 26], [56.5, 25.5],
        [56, 25.5], [56, 26], [56, 26.5]
    ]
}

SUEZ_CANAL = {
    "name": "Suez Canal",
    "type": "canal",
    "polygon": [
        [32.3, 31.3], [32.35, 31.2], [32.35, 30.0], [32.3, 29.9],
        [32.25, 29.9], [32.25, 31.2], [32.3, 31.3]
    ]
}

PANAMA_CANAL = {
    "name": "Panama Canal",
    "type": "canal",
    "polygon": [
        [-79.9, 9.4], [-79.5, 9.4], [-79.5, 9.0], [-79.9, 9.0], [-79.9, 9.4]
    ]
}

# ============================================================
# REGIONAL GROUPINGS
# ============================================================

ALL_WATER_BODIES = [
    # Oceans
    PACIFIC_OCEAN,
    ATLANTIC_OCEAN,
    INDIAN_OCEAN,
    ARCTIC_OCEAN,
    SOUTHERN_OCEAN,
    
    # Atlantic Seas
    CARIBBEAN_SEA,
    GULF_OF_MEXICO,
    MEDITERRANEAN_SEA,
    BLACK_SEA,
    NORTH_SEA,
    BALTIC_SEA,
    
    # Pacific Seas
    SOUTH_CHINA_SEA,
    EAST_CHINA_SEA,
    SEA_OF_JAPAN,
    BERING_SEA,
    
    # Indian Ocean Seas
    ARABIAN_SEA,
    BAY_OF_BENGAL,
    PERSIAN_GULF,
    RED_SEA,
    
    # Straits and Canals
    STRAIT_OF_GIBRALTAR,
    STRAIT_OF_MALACCA,
    STRAIT_OF_HORMUZ,
    SUEZ_CANAL,
    PANAMA_CANAL
]

# Create lookup dictionaries
WATER_BODIES_BY_NAME = {wb["name"]: wb for wb in ALL_WATER_BODIES}
WATER_BODIES_BY_TYPE = {}
for wb in ALL_WATER_BODIES:
    wb_type = wb["type"]
    if wb_type not in WATER_BODIES_BY_TYPE:
        WATER_BODIES_BY_TYPE[wb_type] = []
    WATER_BODIES_BY_TYPE[wb_type].append(wb)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def point_in_polygon(lon, lat, polygon):
    """
    Check if a point (lon, lat) is inside a polygon.
    Uses ray-casting algorithm.
    
    Args:
        lon: Longitude of point
        lat: Latitude of point
        polygon: List of [lon, lat] coordinate pairs
        
    Returns:
        True if point is inside polygon, False otherwise
    """
    n = len(polygon)
    inside = False
    
    p1_lon, p1_lat = polygon[0]
    for i in range(1, n + 1):
        p2_lon, p2_lat = polygon[i % n]
        if lat > min(p1_lat, p2_lat):
            if lat <= max(p1_lat, p2_lat):
                if lon <= max(p1_lon, p2_lon):
                    if p1_lat != p2_lat:
                        x_intersection = (lat - p1_lat) * (p2_lon - p1_lon) / (p2_lat - p1_lat) + p1_lon
                    if p1_lon == p2_lon or lon <= x_intersection:
                        inside = not inside
        p1_lon, p1_lat = p2_lon, p2_lat
    
    return inside


def identify_water_body(lon, lat):
    """
    Identify which water body a point is in.
    Checks from most specific (straits/canals) to least specific (oceans).
    
    Args:
        lon: Longitude of point
        lat: Latitude of point
        
    Returns:
        Water body dictionary or None if not found
    """
    # Check in order of specificity
    type_order = ["canal", "strait", "gulf", "bay", "sea", "ocean"]
    
    for wb_type in type_order:
        if wb_type in WATER_BODIES_BY_TYPE:
            for wb in WATER_BODIES_BY_TYPE[wb_type]:
                if point_in_polygon(lon, lat, wb["polygon"]):
                    return wb
    
    return None


def get_water_bodies_by_type(water_type):
    """
    Get all water bodies of a specific type.
    
    Args:
        water_type: Type of water body (ocean, sea, gulf, bay, strait, canal)
        
    Returns:
        List of water body dictionaries
    """
    return WATER_BODIES_BY_TYPE.get(water_type, [])


def get_water_body_by_name(name):
    """
    Get a water body by its name.
    
    Args:
        name: Name of the water body
        
    Returns:
        Water body dictionary or None if not found
    """
    return WATER_BODIES_BY_NAME.get(name)


def get_all_water_body_names():
    """
    Get a list of all water body names.
    
    Returns:
        List of water body names
    """
    return list(WATER_BODIES_BY_NAME.keys())


def get_bounding_box(polygon):
    """
    Get the bounding box of a polygon.
    
    Args:
        polygon: List of [lon, lat] coordinate pairs
        
    Returns:
        Tuple of (min_lon, min_lat, max_lon, max_lat)
    """
    lons = [p[0] for p in polygon]
    lats = [p[1] for p in polygon]
    return (min(lons), min(lats), max(lons), max(lats))


def filter_vessels_by_water_body(df, water_body_name):
    """
    Filter a DataFrame of vessels by water body.
    
    Args:
        df: DataFrame with LON and LAT columns
        water_body_name: Name of the water body
        
    Returns:
        Filtered DataFrame
    """
    wb = get_water_body_by_name(water_body_name)
    if wb is None:
        return df.iloc[0:0]  # Empty DataFrame
    
    polygon = wb["polygon"]
    mask = df.apply(lambda row: point_in_polygon(row['LON'], row['LAT'], polygon), axis=1)
    return df[mask]


# ============================================================
# EXPORT FOR FRONTEND
# ============================================================

def export_water_bodies_geojson():
    """
    Export all water bodies as GeoJSON FeatureCollection.
    Useful for map visualization.
    
    Returns:
        GeoJSON dictionary
    """
    features = []
    for wb in ALL_WATER_BODIES:
        # Convert polygon to GeoJSON format (close the ring)
        coordinates = [wb["polygon"] + [wb["polygon"][0]]]
        
        feature = {
            "type": "Feature",
            "properties": {
                "name": wb["name"],
                "type": wb["type"],
                "parent": wb.get("parent", "")
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": coordinates
            }
        }
        features.append(feature)
    
    return {
        "type": "FeatureCollection",
        "features": features
    }


if __name__ == "__main__":
    # Example usage and testing
    import json
    
    print("=" * 60)
    print("WATER BODIES REFERENCE SYSTEM")
    print("=" * 60)
    
    print(f"\nTotal water bodies defined: {len(ALL_WATER_BODIES)}")
    print(f"\nWater bodies by type:")
    for wb_type, bodies in WATER_BODIES_BY_TYPE.items():
        print(f"  {wb_type.capitalize()}: {len(bodies)}")
        for wb in bodies:
            print(f"    - {wb['name']}")
    
    # Test point identification
    print(f"\n{'=' * 60}")
    print("TESTING POINT IDENTIFICATION")
    print("=" * 60)
    
    test_points = [
        (30, 30, "Mediterranean Sea"),
        (-75, 25, "Atlantic Ocean / Caribbean"),
        (120, 15, "South China Sea"),
        (55, 26, "Persian Gulf"),
        (-5.5, 36, "Strait of Gibraltar"),
        (0, 0, "Atlantic Ocean"),
        (150, -30, "Pacific Ocean"),
    ]
    
    for lon, lat, expected in test_points:
        result = identify_water_body(lon, lat)
        if result:
            print(f"\n[OK] ({lon}, {lat}) -> {result['name']}")
        else:
            print(f"\n[MISS] ({lon}, {lat}) -> Not found (expected: {expected})")
    
    # Export GeoJSON
    print(f"\n{'=' * 60}")
    print("EXPORTING GEOJSON")
    print("=" * 60)
    geojson = export_water_bodies_geojson()
    print(f"\nGeoJSON features: {len(geojson['features'])}")
    print("Save to file: water_bodies.geojson")


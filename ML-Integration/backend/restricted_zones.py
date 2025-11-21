"""
Restricted Naval Zones - Geographic Definitions

Defines military restricted areas, marine protected areas, piracy risk zones,
sanctioned regions, and other monitored maritime zones as simplified lat/lon polygons
for efficient zone violation detection.

Each zone is defined as a list of [longitude, latitude] coordinate pairs.
Polygons are simplified to use fewer points while maintaining reasonable accuracy.

Coordinate format: [longitude, latitude] (standard GeoJSON format)
- Longitude: -180 to 180 (negative = West, positive = East)
- Latitude: -90 to 90 (negative = South, positive = North)
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# ============================================================
# PIRACY RISK ZONES
# ============================================================

GULF_OF_ADEN_HRA = {
    "name": "Gulf of Aden High Risk Area (HRA)",
    "type": "piracy_risk",
    "risk_level": "high",
    "description": "Internationally recognized high-risk area for piracy",
    "polygon": [
        (43, 11), (51, 11), (54, 13), (54, 16),
        (51, 16), (47, 14), (43, 13), (43, 11)
    ],
    "notes": "Somali piracy risk zone. Armed guards recommended."
}

HORN_OF_AFRICA_HRA = {
    "name": "Horn of Africa Extended HRA",
    "type": "piracy_risk",
    "risk_level": "high",
    "description": "Extended piracy risk area off Somalia",
    "polygon": [
        (40, 0), (65, 0), (78, 5), (78, 20),
        (60, 20), (51, 16), (43, 11), (40, 5), (40, 0)
    ],
    "notes": "Extended risk area for Somali piracy operations"
}

STRAIT_OF_MALACCA_PIRACY = {
    "name": "Strait of Malacca Piracy Zone",
    "type": "piracy_risk",
    "risk_level": "medium",
    "description": "Piracy and armed robbery hotspot",
    "polygon": [
        (98, 1), (104, 1), (104, 6), (99, 7), (98, 5), (98, 1)
    ],
    "notes": "Armed robbery, theft, and occasional hijackings"
}

SINGAPORE_STRAITS_RISK = {
    "name": "Singapore Straits Risk Area",
    "type": "piracy_risk",
    "risk_level": "medium",
    "description": "Armed robbery zone in busy shipping lane",
    "polygon": [
        (103, 1), (104.5, 1), (104.5, 1.5), (103, 1.5), (103, 1)
    ],
    "notes": "Opportunistic theft from anchored vessels"
}

GULF_OF_GUINEA_HRA = {
    "name": "Gulf of Guinea High Risk Area",
    "type": "piracy_risk",
    "risk_level": "high",
    "description": "Most dangerous piracy zone globally",
    "polygon": [
        (-10, -5), (10, -5), (10, 5), (5, 8),
        (-5, 8), (-10, 5), (-10, -5)
    ],
    "notes": "Kidnapping for ransom, violent attacks common. Armed guards recommended."
}

# ============================================================
# SANCTIONED AREAS & PORTS
# ============================================================

NORTH_KOREA_SANCTIONS_ZONE = {
    "name": "North Korea Sanctions Enforcement Zone",
    "type": "sanctions",
    "risk_level": "critical",
    "description": "UN sanctions enforcement area",
    "polygon": [
        (124, 37), (131, 37), (131, 43), (129, 43),
        (127, 42), (125, 41), (124, 39), (124, 37)
    ],
    "notes": "UN sanctions prohibit trade. Ship-to-ship transfers monitored."
}

CRIMEA_SANCTIONS_ZONE = {
    "name": "Crimea Sanctions Zone",
    "type": "sanctions",
    "risk_level": "high",
    "description": "International sanctions on Crimean ports",
    "polygon": [
        (32, 44), (36, 44), (36, 46), (33, 46), (32, 45), (32, 44)
    ],
    "notes": "EU/US sanctions on Crimean ports (Sevastopol, Kerch, etc.)"
}

IRAN_SANCTIONS_ZONE = {
    "name": "Iran Sanctions Monitoring Zone",
    "type": "sanctions",
    "risk_level": "high",
    "description": "Iranian waters sanctions monitoring",
    "polygon": [
        (48, 24), (61, 24), (61, 31), (56, 38),
        (50, 38), (48, 30), (48, 24)
    ],
    "notes": "US/international sanctions. Oil export monitoring."
}

VENEZUELA_SANCTIONS_PORTS = {
    "name": "Venezuela Sanctions Zone",
    "type": "sanctions",
    "risk_level": "high",
    "description": "Venezuelan sanctioned ports",
    "polygon": [
        (-73, 8), (-60, 8), (-60, 12), (-70, 12), (-73, 10), (-73, 8)
    ],
    "notes": "US sanctions on Venezuelan oil exports"
}

# ============================================================
# CONTESTED WATERS & DISPUTED ZONES
# ============================================================

SOUTH_CHINA_SEA_DISPUTED = {
    "name": "South China Sea Disputed Waters",
    "type": "contested",
    "risk_level": "high",
    "description": "Territorial disputes, military activity",
    "polygon": [
        (105, 3), (120, 3), (122, 8), (119, 12),
        (115, 20), (112, 22), (108, 20), (105, 15), (105, 3)
    ],
    "notes": "Nine-dash line disputes. Multiple claimants. Military presence."
}

SPRATLY_ISLANDS_ZONE = {
    "name": "Spratly Islands Exclusion Zone",
    "type": "contested",
    "risk_level": "critical",
    "description": "Heavily militarized disputed islands",
    "polygon": [
        (111, 7), (117, 7), (117, 12), (111, 12), (111, 7)
    ],
    "notes": "Artificial islands, military installations, restricted access"
}

PARACEL_ISLANDS_ZONE = {
    "name": "Paracel Islands Restricted Zone",
    "type": "contested",
    "risk_level": "critical",
    "description": "Chinese-controlled disputed territory",
    "polygon": [
        (110, 15), (113, 15), (113, 17), (110, 17), (110, 15)
    ],
    "notes": "Restricted by China. Disputed by Vietnam."
}

SENKAKU_DIAOYU_ZONE = {
    "name": "Senkaku/Diaoyu Islands Zone",
    "type": "contested",
    "risk_level": "high",
    "description": "Japan-China disputed islands",
    "polygon": [
        (123.3, 25.6), (124.5, 25.6), (124.5, 26.0), (123.3, 26.0), (123.3, 25.6)
    ],
    "notes": "Patrolled by Japan Coast Guard. Claimed by China."
}

KERCH_STRAIT_ZONE = {
    "name": "Kerch Strait Restricted Area",
    "type": "contested",
    "risk_level": "critical",
    "description": "Crimea-Russia strait, restricted by Russia",
    "polygon": [
        (35.5, 45.0), (36.7, 45.0), (36.7, 45.4), (35.5, 45.4), (35.5, 45.0)
    ],
    "notes": "Russia controls access. International disputes."
}

# ============================================================
# MILITARY RESTRICTED AREAS
# ============================================================

DIEGO_GARCIA_EXCLUSION = {
    "name": "Diego Garcia Naval Base Exclusion Zone",
    "type": "military",
    "risk_level": "critical",
    "description": "US/UK military base, restricted access",
    "polygon": [
        (72.3, -7.4), (72.5, -7.4), (72.5, -7.2), (72.3, -7.2), (72.3, -7.4)
    ],
    "notes": "Major US naval base. No unauthorized vessels permitted."
}

NORFOLK_NAVAL_RESTRICTED = {
    "name": "Norfolk Naval Base Security Zone",
    "type": "military",
    "risk_level": "high",
    "description": "World's largest naval base restricted area",
    "polygon": [
        (-76.4, 36.8), (-76.2, 36.8), (-76.2, 37.0), (-76.4, 37.0), (-76.4, 36.8)
    ],
    "notes": "US Atlantic Fleet HQ. Strict access controls."
}

GUAM_NAVAL_ZONE = {
    "name": "Guam Naval Security Zone",
    "type": "military",
    "risk_level": "high",
    "description": "US naval base restricted waters",
    "polygon": [
        (144.6, 13.4), (144.7, 13.4), (144.7, 13.5), (144.6, 13.5), (144.6, 13.4)
    ],
    "notes": "Strategic Pacific naval base"
}

YOKOSUKA_NAVAL_BASE = {
    "name": "Yokosuka Naval Base Zone",
    "type": "military",
    "risk_level": "medium",
    "description": "US 7th Fleet headquarters",
    "polygon": [
        (139.6, 35.2), (139.7, 35.2), (139.7, 35.3), (139.6, 35.3), (139.6, 35.2)
    ],
    "notes": "US naval presence in Japan"
}

KINGS_BAY_SUBMARINE_BASE = {
    "name": "Kings Bay Submarine Base Zone",
    "type": "military",
    "risk_level": "high",
    "description": "US submarine base restricted area",
    "polygon": [
        (-81.6, 30.7), (-81.5, 30.7), (-81.5, 30.8), (-81.6, 30.8), (-81.6, 30.7)
    ],
    "notes": "Nuclear submarine operations"
}

# ============================================================
# MARINE PROTECTED AREAS (CRITICAL)
# ============================================================

GALAPAGOS_MARINE_RESERVE = {
    "name": "Galápagos Marine Reserve",
    "type": "marine_protected",
    "risk_level": "critical",
    "description": "UNESCO World Heritage Site, fishing prohibited",
    "polygon": [
        (-92, -2), (-89, -2), (-89, 1), (-92, 1), (-92, -2)
    ],
    "notes": "Illegal fishing common. Heavy fines. Ecuadorian enforcement."
}

GREAT_BARRIER_REEF_ZONE = {
    "name": "Great Barrier Reef Marine Park",
    "type": "marine_protected",
    "risk_level": "high",
    "description": "World's largest coral reef system, protected",
    "polygon": [
        (142, -24), (154, -24), (154, -10), (145, -10), (142, -15), (142, -24)
    ],
    "notes": "Strict vessel restrictions. No anchoring. Speed limits."
}

PHOENIX_ISLANDS_PROTECTED_AREA = {
    "name": "Phoenix Islands Protected Area",
    "type": "marine_protected",
    "risk_level": "high",
    "description": "One of world's largest marine protected areas",
    "polygon": [
        (-177, -6), (-169, -6), (-169, 0), (-177, 0), (-177, -6)
    ],
    "notes": "Kiribati. Fishing banned. Transit allowed."
}

# ============================================================
# STRAITS & CHOKEPOINTS (MONITORED)
# ============================================================

STRAIT_OF_HORMUZ_MONITORED = {
    "name": "Strait of Hormuz Monitoring Zone",
    "type": "strategic_chokepoint",
    "risk_level": "critical",
    "description": "20% of world's oil passes through",
    "polygon": [
        (56.0, 25.5), (57.0, 25.5), (57.0, 26.5), (56.0, 26.5), (56.0, 25.5)
    ],
    "notes": "Iran-Oman. Heavy military presence. Tensions high."
}

BAB_EL_MANDEB_MONITORED = {
    "name": "Bab-el-Mandeb Strait Monitoring Zone",
    "type": "strategic_chokepoint",
    "risk_level": "high",
    "description": "Red Sea-Gulf of Aden connection",
    "polygon": [
        (42.5, 12.5), (43.5, 12.5), (43.5, 13.0), (42.5, 13.0), (42.5, 12.5)
    ],
    "notes": "Yemen instability. Houthi attacks on shipping. Piracy risk."
}

TAIWAN_STRAIT_MONITORED = {
    "name": "Taiwan Strait Monitoring Zone",
    "type": "strategic_chokepoint",
    "risk_level": "critical",
    "description": "China-Taiwan tensions, military activity",
    "polygon": [
        (118, 23), (120, 23), (120, 26), (118, 26), (118, 23)
    ],
    "notes": "Heavy military presence. Political tensions."
}

# ============================================================
# FISHING RESTRICTED ZONES
# ============================================================

NORTH_ATLANTIC_FISHING_CLOSURE = {
    "name": "North Atlantic Seasonal Closure",
    "type": "fishing_restricted",
    "risk_level": "medium",
    "description": "Seasonal fishing restrictions for conservation",
    "polygon": [
        (-45, 40), (-35, 40), (-35, 50), (-45, 50), (-45, 40)
    ],
    "notes": "Varies by season. Check NAFO regulations."
}

# ============================================================
# REGIONAL GROUPINGS
# ============================================================

ALL_RESTRICTED_ZONES = [
    # Piracy Risk Zones
    GULF_OF_ADEN_HRA,
    HORN_OF_AFRICA_HRA,
    STRAIT_OF_MALACCA_PIRACY,
    SINGAPORE_STRAITS_RISK,
    GULF_OF_GUINEA_HRA,
    
    # Sanctioned Areas
    NORTH_KOREA_SANCTIONS_ZONE,
    CRIMEA_SANCTIONS_ZONE,
    IRAN_SANCTIONS_ZONE,
    VENEZUELA_SANCTIONS_PORTS,
    
    # Contested Waters
    SOUTH_CHINA_SEA_DISPUTED,
    SPRATLY_ISLANDS_ZONE,
    PARACEL_ISLANDS_ZONE,
    SENKAKU_DIAOYU_ZONE,
    KERCH_STRAIT_ZONE,
    
    # Military Restricted
    DIEGO_GARCIA_EXCLUSION,
    NORFOLK_NAVAL_RESTRICTED,
    GUAM_NAVAL_ZONE,
    YOKOSUKA_NAVAL_BASE,
    KINGS_BAY_SUBMARINE_BASE,
    
    # Marine Protected
    GALAPAGOS_MARINE_RESERVE,
    GREAT_BARRIER_REEF_ZONE,
    PHOENIX_ISLANDS_PROTECTED_AREA,
    
    # Strategic Chokepoints
    STRAIT_OF_HORMUZ_MONITORED,
    BAB_EL_MANDEB_MONITORED,
    TAIWAN_STRAIT_MONITORED,
    
    # Fishing Restricted
    NORTH_ATLANTIC_FISHING_CLOSURE,
]

# Create lookup dictionaries
RESTRICTED_ZONES_BY_NAME = {zone["name"]: zone for zone in ALL_RESTRICTED_ZONES}

RESTRICTED_ZONES_BY_TYPE = {}
for zone in ALL_RESTRICTED_ZONES:
    zone_type = zone["type"]
    if zone_type not in RESTRICTED_ZONES_BY_TYPE:
        RESTRICTED_ZONES_BY_TYPE[zone_type] = []
    RESTRICTED_ZONES_BY_TYPE[zone_type].append(zone)

RESTRICTED_ZONES_BY_RISK = {}
for zone in ALL_RESTRICTED_ZONES:
    risk_level = zone.get("risk_level", "medium")
    if risk_level not in RESTRICTED_ZONES_BY_RISK:
        RESTRICTED_ZONES_BY_RISK[risk_level] = []
    RESTRICTED_ZONES_BY_RISK[risk_level].append(zone)


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
        polygon: List of (lon, lat) tuples
        
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


def identify_restricted_zone(lon, lat):
    """
    Identify which restricted zone(s) a point is in.
    A point may be in multiple zones (e.g., a piracy zone AND a contested area).
    
    Args:
        lon: Longitude of point
        lat: Latitude of point
        
    Returns:
        List of restricted zone dictionaries (may be empty if not in any zone)
    """
    zones_found = []
    
    for zone in ALL_RESTRICTED_ZONES:
        if point_in_polygon(lon, lat, zone["polygon"]):
            zones_found.append(zone)
    
    return zones_found


def get_zones_by_type(zone_type):
    """
    Get all restricted zones of a specific type.
    
    Args:
        zone_type: Type of zone (piracy_risk, sanctions, contested, military, marine_protected, strategic_chokepoint, fishing_restricted)
        
    Returns:
        List of zone dictionaries
    """
    return RESTRICTED_ZONES_BY_TYPE.get(zone_type, [])


def get_zones_by_risk_level(risk_level):
    """
    Get all zones of a specific risk level.
    
    Args:
        risk_level: Risk level (critical, high, medium, low)
        
    Returns:
        List of zone dictionaries
    """
    return RESTRICTED_ZONES_BY_RISK.get(risk_level, [])


def get_zone_by_name(name):
    """
    Get a restricted zone by its name.
    
    Args:
        name: Name of the zone
        
    Returns:
        Zone dictionary or None if not found
    """
    return RESTRICTED_ZONES_BY_NAME.get(name)


def get_all_zone_names():
    """
    Get a list of all restricted zone names.
    
    Returns:
        List of zone names
    """
    return list(RESTRICTED_ZONES_BY_NAME.keys())


def filter_vessels_by_restricted_zone(df, zone_name):
    """
    Filter a DataFrame of vessels by restricted zone.
    
    Args:
        df: DataFrame with LON and LAT columns
        zone_name: Name of the restricted zone
        
    Returns:
        Filtered DataFrame
    """
    zone = get_zone_by_name(zone_name)
    if zone is None:
        return df.iloc[0:0]  # Empty DataFrame
    
    polygon = zone["polygon"]
    mask = df.apply(lambda row: point_in_polygon(row['LON'], row['LAT'], polygon), axis=1)
    return df[mask]


def check_zone_violation(lon, lat):
    """
    Check if a location violates any restricted zones.
    
    Args:
        lon: Longitude
        lat: Latitude
        
    Returns:
        Dictionary with violation details
    """
    zones = identify_restricted_zone(lon, lat)
    
    if not zones:
        return {
            "violation": False,
            "location": {"longitude": lon, "latitude": lat},
            "zones": []
        }
    
    return {
        "violation": True,
        "location": {"longitude": lon, "latitude": lat},
        "zones": [
            {
                "name": z["name"],
                "type": z["type"],
                "risk_level": z.get("risk_level", "unknown"),
                "description": z["description"],
                "notes": z.get("notes", "")
            }
            for z in zones
        ],
        "highest_risk": max([z.get("risk_level", "low") for z in zones], 
                           key=lambda x: {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(x, 0))
    }


def get_zone_stats():
    """
    Get statistics about defined restricted zones.
    
    Returns:
        Dictionary with counts by type and risk level
    """
    stats = {
        "total": len(ALL_RESTRICTED_ZONES),
        "by_type": {},
        "by_risk_level": {}
    }
    
    for zone_type, zones in RESTRICTED_ZONES_BY_TYPE.items():
        stats["by_type"][zone_type] = {
            "count": len(zones),
            "names": [z["name"] for z in zones]
        }
    
    for risk_level, zones in RESTRICTED_ZONES_BY_RISK.items():
        stats["by_risk_level"][risk_level] = {
            "count": len(zones),
            "names": [z["name"] for z in zones]
        }
    
    return stats


# ============================================================
# EXPORT FOR FRONTEND
# ============================================================

def export_restricted_zones_geojson():
    """
    Export all restricted zones as GeoJSON FeatureCollection.
    Useful for map visualization with color coding by risk level.
    
    Returns:
        GeoJSON dictionary
    """
    features = []
    
    # Color coding by risk level
    risk_colors = {
        "critical": "#ff0000",  # Red
        "high": "#ff6600",      # Orange
        "medium": "#ffcc00",    # Yellow
        "low": "#00ff00"        # Green
    }
    
    for zone in ALL_RESTRICTED_ZONES:
        # Convert polygon to GeoJSON format (close the ring)
        polygon_coords = list(zone["polygon"])
        if polygon_coords[0] != polygon_coords[-1]:
            polygon_coords.append(polygon_coords[0])
        coordinates = [polygon_coords]
        
        feature = {
            "type": "Feature",
            "properties": {
                "name": zone["name"],
                "type": zone["type"],
                "risk_level": zone.get("risk_level", "medium"),
                "description": zone["description"],
                "notes": zone.get("notes", ""),
                "color": risk_colors.get(zone.get("risk_level", "medium"), "#ffcc00")
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
    print("RESTRICTED NAVAL ZONES REFERENCE SYSTEM")
    print("=" * 60)
    
    stats = get_zone_stats()
    print(f"\nTotal restricted zones defined: {stats['total']}")
    
    print(f"\nZones by type:")
    for zone_type, info in stats['by_type'].items():
        print(f"  {zone_type.replace('_', ' ').title()}: {info['count']}")
        for name in info['names']:
            print(f"    - {name}")
    
    print(f"\nZones by risk level:")
    for risk_level, info in stats['by_risk_level'].items():
        print(f"  {risk_level.upper()}: {info['count']}")
        for name in info['names']:
            print(f"    - {name}")
    
    # Test zone violation checks
    print(f"\n{'=' * 60}")
    print("TESTING ZONE VIOLATION DETECTION")
    print("=" * 60)
    
    test_points = [
        (56.5, 26, "Strait of Hormuz"),
        (120, 15, "South China Sea"),
        (45, 12, "Gulf of Aden"),
        (0, 0, "Safe waters"),
        (144.65, 13.45, "Guam Naval Base"),
    ]
    
    for lon, lat, description in test_points:
        result = check_zone_violation(lon, lat)
        if result["violation"]:
            print(f"\n[VIOLATION] ({lon}, {lat}) - {description}")
            for zone in result["zones"]:
                print(f"  -> {zone['name']} ({zone['risk_level'].upper()} risk)")
                print(f"     Type: {zone['type']}")
                print(f"     {zone['description']}")
        else:
            print(f"\n[OK] ({lon}, {lat}) - {description}: No violations")
    
    # Export GeoJSON
    print(f"\n{'=' * 60}")
    print("EXPORTING GEOJSON")
    print("=" * 60)
    geojson = export_restricted_zones_geojson()
    print(f"\nGeoJSON features: {len(geojson['features'])}")
    print("Save to file: restricted_zones.geojson")


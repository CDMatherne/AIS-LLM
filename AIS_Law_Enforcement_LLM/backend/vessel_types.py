"""
Vessel Types - AIS Ship Type Code Definitions

Defines standard AIS ship type codes (20-99) with names, categories,
and filtering capabilities for vessel-specific analysis.

Based on AIS standard ship type classification used in maritime data.
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# ============================================================
# VESSEL TYPE DEFINITIONS
# Based on AIS Standard Ship Type Codes (20-99)
# ============================================================

VESSEL_TYPES = {
    # Wing in ground (WIG) - 20-29
    20: {'name': 'Wing in ground (WIG), all ships of this type', 'category': 'WIG', 'description': 'Ground-effect vehicles'},
    21: {'name': 'Wing in ground (WIG), Hazardous category A', 'category': 'WIG', 'description': 'WIG carrying hazardous materials category A'},
    22: {'name': 'Wing in ground (WIG), Hazardous category B', 'category': 'WIG', 'description': 'WIG carrying hazardous materials category B'},
    23: {'name': 'Wing in ground (WIG), Hazardous category C', 'category': 'WIG', 'description': 'WIG carrying hazardous materials category C'},
    24: {'name': 'Wing in ground (WIG), Hazardous category D', 'category': 'WIG', 'description': 'WIG carrying hazardous materials category D'},
    
    # Special craft - 30-39
    30: {'name': 'Fishing', 'category': 'Special', 'description': 'Commercial fishing vessels'},
    31: {'name': 'Towing', 'category': 'Special', 'description': 'Towing vessels'},
    32: {'name': 'Towing: length exceeds 200m or breadth exceeds 25m', 'category': 'Special', 'description': 'Large towing operations'},
    33: {'name': 'Dredging or underwater ops', 'category': 'Special', 'description': 'Dredging or underwater operations'},
    34: {'name': 'Diving ops', 'category': 'Special', 'description': 'Diving support vessels'},
    35: {'name': 'Military ops', 'category': 'Special', 'description': 'Military operations vessels'},
    36: {'name': 'Sailing', 'category': 'Special', 'description': 'Sailing vessels'},
    37: {'name': 'Pleasure Craft', 'category': 'Special', 'description': 'Recreational pleasure craft'},
    38: {'name': 'Reserved', 'category': 'Special', 'description': 'Reserved ship type'},
    39: {'name': 'Reserved', 'category': 'Special', 'description': 'Reserved ship type'},
    
    # High speed craft (HSC) - 40-49
    40: {'name': 'High speed craft (HSC), all ships of this type', 'category': 'HSC', 'description': 'High-speed craft (HSC)'},
    41: {'name': 'High speed craft (HSC), Hazardous category A', 'category': 'HSC', 'description': 'HSC carrying hazardous materials category A'},
    42: {'name': 'High speed craft (HSC), Hazardous category B', 'category': 'HSC', 'description': 'HSC carrying hazardous materials category B'},
    43: {'name': 'High speed craft (HSC), Hazardous category C', 'category': 'HSC', 'description': 'HSC carrying hazardous materials category C'},
    44: {'name': 'High speed craft (HSC), Hazardous category D', 'category': 'HSC', 'description': 'HSC carrying hazardous materials category D'},
    45: {'name': 'High speed craft (HSC), Reserved', 'category': 'HSC', 'description': 'HSC reserved type'},
    46: {'name': 'High speed craft (HSC), Reserved', 'category': 'HSC', 'description': 'HSC reserved type'},
    47: {'name': 'High speed craft (HSC), Reserved', 'category': 'HSC', 'description': 'HSC reserved type'},
    48: {'name': 'High speed craft (HSC), Reserved', 'category': 'HSC', 'description': 'HSC reserved type'},
    49: {'name': 'High speed craft (HSC), No additional information', 'category': 'HSC', 'description': 'HSC without specific subtype'},
    
    # Special purpose - 50-59
    50: {'name': 'Pilot Vessel', 'category': 'Special Purpose', 'description': 'Harbor pilot vessels'},
    51: {'name': 'Search and Rescue vessel', 'category': 'Special Purpose', 'description': 'SAR vessels'},
    52: {'name': 'Tug', 'category': 'Special Purpose', 'description': 'Tugboats'},
    53: {'name': 'Port Tender', 'category': 'Special Purpose', 'description': 'Port service vessels'},
    54: {'name': 'Anti-pollution equipment', 'category': 'Special Purpose', 'description': 'Environmental response vessels'},
    55: {'name': 'Law Enforcement', 'category': 'Special Purpose', 'description': 'Coast guard, police vessels'},
    56: {'name': 'Spare - Local Vessel', 'category': 'Special Purpose', 'description': 'Local/spare designation'},
    57: {'name': 'Spare - Local Vessel', 'category': 'Special Purpose', 'description': 'Local/spare designation'},
    58: {'name': 'Medical Transport', 'category': 'Special Purpose', 'description': 'Hospital/ambulance ships'},
    59: {'name': 'Noncombatant ship (RR Resolution No. 18)', 'category': 'Special Purpose', 'description': 'Noncombatant vessels'},
    
    # Passenger - 60-69
    60: {'name': 'Passenger, all ships of this type', 'category': 'Passenger', 'description': 'Passenger vessels'},
    61: {'name': 'Passenger, Hazardous category A', 'category': 'Passenger', 'description': 'Passenger ships carrying hazardous materials category A'},
    62: {'name': 'Passenger, Hazardous category B', 'category': 'Passenger', 'description': 'Passenger ships carrying hazardous materials category B'},
    63: {'name': 'Passenger, Hazardous category C', 'category': 'Passenger', 'description': 'Passenger ships carrying hazardous materials category C'},
    64: {'name': 'Passenger, Hazardous category D', 'category': 'Passenger', 'description': 'Passenger ships carrying hazardous materials category D'},
    69: {'name': 'Passenger, No additional information', 'category': 'Passenger', 'description': 'Passenger ships without specific subtype'},
    
    # Cargo - 70-79
    70: {'name': 'Cargo, all ships of this type', 'category': 'Cargo', 'description': 'Cargo vessels'},
    71: {'name': 'Cargo, Hazardous category A', 'category': 'Cargo', 'description': 'Cargo ships carrying hazardous materials category A'},
    72: {'name': 'Cargo, Hazardous category B', 'category': 'Cargo', 'description': 'Cargo ships carrying hazardous materials category B'},
    73: {'name': 'Cargo, Hazardous category C', 'category': 'Cargo', 'description': 'Cargo ships carrying hazardous materials category C'},
    74: {'name': 'Cargo, Hazardous category D', 'category': 'Cargo', 'description': 'Cargo ships carrying hazardous materials category D'},
    79: {'name': 'Cargo, No additional information', 'category': 'Cargo', 'description': 'Cargo ships without specific subtype'},
    
    # Tanker - 80-89
    80: {'name': 'Tanker, all ships of this type', 'category': 'Tanker', 'description': 'Oil/chemical tankers'},
    81: {'name': 'Tanker, Hazardous category A', 'category': 'Tanker', 'description': 'Tankers carrying hazardous materials category A'},
    82: {'name': 'Tanker, Hazardous category B', 'category': 'Tanker', 'description': 'Tankers carrying hazardous materials category B'},
    83: {'name': 'Tanker, Hazardous category C', 'category': 'Tanker', 'description': 'Tankers carrying hazardous materials category C'},
    84: {'name': 'Tanker, Hazardous category D', 'category': 'Tanker', 'description': 'Tankers carrying hazardous materials category D'},
    89: {'name': 'Tanker, No additional information', 'category': 'Tanker', 'description': 'Tankers without specific subtype'},
    
    # Other - 90-99
    90: {'name': 'Other Type, all ships of this type', 'category': 'Other', 'description': 'Other vessel types'},
    91: {'name': 'Other Type, Hazardous category A', 'category': 'Other', 'description': 'Other vessels carrying hazardous materials category A'},
    92: {'name': 'Other Type, Hazardous category B', 'category': 'Other', 'description': 'Other vessels carrying hazardous materials category B'},
    93: {'name': 'Other Type, Hazardous category C', 'category': 'Other', 'description': 'Other vessels carrying hazardous materials category C'},
    94: {'name': 'Other Type, Hazardous category D', 'category': 'Other', 'description': 'Other vessels carrying hazardous materials category D'},
}

# Create lookup dictionaries
VESSEL_TYPES_BY_CATEGORY = {}
for code, details in VESSEL_TYPES.items():
    category = details['category']
    if category not in VESSEL_TYPES_BY_CATEGORY:
        VESSEL_TYPES_BY_CATEGORY[category] = []
    VESSEL_TYPES_BY_CATEGORY[category].append(code)

# Category ranges for quick filtering
CATEGORY_RANGES = {
    'WIG': (20, 29),
    'Special': (30, 39),
    'HSC': (40, 49),
    'Special Purpose': (50, 59),
    'Passenger': (60, 69),
    'Cargo': (70, 79),
    'Tanker': (80, 89),
    'Other': (90, 99)
}

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def get_vessel_type_name(vessel_type_code):
    """
    Get the name of a vessel type by its code.
    
    Args:
        vessel_type_code: AIS vessel type code (20-99)
        
    Returns:
        Vessel type name or 'Unknown' if not found
    """
    if vessel_type_code in VESSEL_TYPES:
        return VESSEL_TYPES[vessel_type_code]['name']
    return f'Unknown ({vessel_type_code})'


def get_vessel_type_category(vessel_type_code):
    """
    Get the category of a vessel type.
    
    Args:
        vessel_type_code: AIS vessel type code (20-99)
        
    Returns:
        Category name or 'Unknown'
    """
    if vessel_type_code in VESSEL_TYPES:
        return VESSEL_TYPES[vessel_type_code]['category']
    return 'Unknown'


def get_vessel_types_by_category(category):
    """
    Get all vessel type codes for a category.
    
    Args:
        category: Category name (WIG, Special, HSC, Special Purpose, Passenger, Cargo, Tanker, Other)
        
    Returns:
        List of vessel type codes
    """
    return VESSEL_TYPES_BY_CATEGORY.get(category, [])


def get_all_categories():
    """
    Get list of all vessel categories.
    
    Returns:
        List of category names
    """
    return list(VESSEL_TYPES_BY_CATEGORY.keys())


def filter_dataframe_by_vessel_type(df, vessel_type_codes):
    """
    Filter a DataFrame by vessel type codes.
    
    Args:
        df: DataFrame with VesselType column
        vessel_type_codes: List of vessel type codes to include
        
    Returns:
        Filtered DataFrame
    """
    if 'VesselType' not in df.columns:
        logger.warning("DataFrame does not have VesselType column")
        return df
    
    return df[df['VesselType'].isin(vessel_type_codes)]


def filter_dataframe_by_category(df, categories):
    """
    Filter a DataFrame by vessel categories.
    
    Args:
        df: DataFrame with VesselType column
        categories: List of category names
        
    Returns:
        Filtered DataFrame
    """
    if 'VesselType' not in df.columns:
        logger.warning("DataFrame does not have VesselType column")
        return df
    
    # Get all vessel type codes for the specified categories
    vessel_type_codes = []
    for category in categories:
        vessel_type_codes.extend(get_vessel_types_by_category(category))
    
    return filter_dataframe_by_vessel_type(df, vessel_type_codes)


def get_vessel_type_stats():
    """
    Get statistics about defined vessel types.
    
    Returns:
        Dictionary with counts by category
    """
    stats = {
        "total": len(VESSEL_TYPES),
        "by_category": {}
    }
    
    for category, codes in VESSEL_TYPES_BY_CATEGORY.items():
        stats["by_category"][category] = {
            "count": len(codes),
            "codes": sorted(codes),
            "range": CATEGORY_RANGES.get(category, (0, 0))
        }
    
    return stats


def get_vessel_type_details(vessel_type_code):
    """
    Get full details about a vessel type.
    
    Args:
        vessel_type_code: AIS vessel type code
        
    Returns:
        Dictionary with vessel type details or None
    """
    if vessel_type_code in VESSEL_TYPES:
        details = VESSEL_TYPES[vessel_type_code].copy()
        details['code'] = vessel_type_code
        return details
    return None


def is_hazardous_cargo(vessel_type_code):
    """
    Check if a vessel type carries hazardous cargo.
    
    Args:
        vessel_type_code: AIS vessel type code
        
    Returns:
        Boolean indicating if vessel carries hazardous materials
    """
    if vessel_type_code in VESSEL_TYPES:
        name = VESSEL_TYPES[vessel_type_code]['name']
        return 'Hazardous' in name
    return False


def is_commercial_vessel(vessel_type_code):
    """
    Check if a vessel is commercial (cargo/tanker).
    
    Args:
        vessel_type_code: AIS vessel type code
        
    Returns:
        Boolean indicating if vessel is commercial
    """
    return vessel_type_code in range(70, 90)


def is_special_purpose_vessel(vessel_type_code):
    """
    Check if a vessel is special purpose (law enforcement, pilot, SAR, etc.).
    
    Args:
        vessel_type_code: AIS vessel type code
        
    Returns:
        Boolean indicating if vessel is special purpose
    """
    return vessel_type_code in range(50, 60)


# ============================================================
# EXPORT FOR INTEGRATION
# ============================================================

if __name__ == "__main__":
    # Example usage and testing
    print("=" * 60)
    print("VESSEL TYPES REFERENCE SYSTEM")
    print("=" * 60)
    
    stats = get_vessel_type_stats()
    print(f"\nTotal vessel types defined: {stats['total']}")
    
    print(f"\nVessel types by category:")
    for category, info in stats['by_category'].items():
        print(f"\n{category}: {info['count']} types (Codes {info['range'][0]}-{info['range'][1]})")
        for code in info['codes'][:3]:  # Show first 3 examples
            vessel_info = VESSEL_TYPES[code]
            print(f"  {code}: {vessel_info['name']}")
        if len(info['codes']) > 3:
            print(f"  ... and {len(info['codes']) - 3} more")
    
    # Test specific queries
    print(f"\n{'=' * 60}")
    print("TESTING VESSEL TYPE QUERIES")
    print("=" * 60)
    
    print(f"\nVessel type 70: {get_vessel_type_name(70)}")
    print(f"Category: {get_vessel_type_category(70)}")
    print(f"Is commercial: {is_commercial_vessel(70)}")
    
    print(f"\nVessel type 55: {get_vessel_type_name(55)}")
    print(f"Category: {get_vessel_type_category(55)}")
    print(f"Is special purpose: {is_special_purpose_vessel(55)}")
    
    print(f"\nVessel type 81: {get_vessel_type_name(81)}")
    print(f"Carries hazardous cargo: {is_hazardous_cargo(81)}")
    
    print(f"\nAll Cargo vessel codes: {get_vessel_types_by_category('Cargo')}")
    print(f"All Tanker vessel codes: {get_vessel_types_by_category('Tanker')}")


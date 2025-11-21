#!/usr/bin/env python3
"""
Diagnostic script to check Python version and import issues
"""

import sys
import os

print("=" * 60)
print("Python Environment Diagnostics")
print("=" * 60)

# Check Python version
print(f"\n1. Python Version: {sys.version}")
print(f"   Python Version Info: {sys.version_info}")
print(f"   Python Executable: {sys.executable}")

# Check if Python 3.8+ (required for importlib.metadata)
if sys.version_info < (3, 8):
    print(f"   ⚠️  WARNING: Python 3.8+ required for importlib.metadata")
else:
    print(f"   ✓ Python version is compatible with importlib.metadata")

# Check Python path
print(f"\n2. Python Path:")
for i, path in enumerate(sys.path, 1):
    print(f"   {i}. {path}")

# Check current working directory
print(f"\n3. Current Working Directory: {os.getcwd()}")

# Check for local files that might shadow standard library
print(f"\n4. Checking for local files that might shadow standard library:")
script_dir = os.path.dirname(os.path.abspath(__file__))
potential_shadows = ['logging.py', 'logging.pyc', 'logging.pyo', '__pycache__/logging.pyc']
for shadow in potential_shadows:
    shadow_path = os.path.join(script_dir, shadow)
    if os.path.exists(shadow_path):
        print(f"   ⚠️  FOUND: {shadow_path} (this could shadow the standard library!)")
    else:
        print(f"   ✓ {shadow} not found")

# Test imports
print(f"\n5. Testing Imports:")

# Test logging import
try:
    import logging
    print(f"   ✓ logging imported successfully")
    print(f"     logging module location: {logging.__file__}")
except ImportError as e:
    print(f"   ✗ FAILED to import logging: {e}")
except Exception as e:
    print(f"   ✗ ERROR importing logging: {e}")

# Test importlib.metadata import
try:
    import importlib.metadata
    print(f"   ✓ importlib.metadata imported successfully")
    print(f"     importlib.metadata module location: {importlib.metadata.__file__}")
except ImportError as e:
    print(f"   ✗ FAILED to import importlib.metadata: {e}")
    print(f"     This requires Python 3.8+. Consider using pkg_resources instead.")
except Exception as e:
    print(f"   ✗ ERROR importing importlib.metadata: {e}")

# Test other critical imports from SFD_GUI.py
print(f"\n6. Testing other imports from SFD_GUI.py:")
test_imports = [
    'os', 'sys', 'configparser', 'tkinter', 'datetime', 
    'subprocess', 'platform', 'time', 're', 'glob', 
    'traceback', 'threading', 'importlib'
]

for module_name in test_imports:
    try:
        __import__(module_name)
        print(f"   ✓ {module_name}")
    except ImportError as e:
        print(f"   ✗ {module_name}: {e}")
    except Exception as e:
        print(f"   ✗ {module_name}: {e}")

# Check if SFD_GUI.py exists
print(f"\n7. Checking for SFD_GUI.py:")
gui_file = os.path.join(script_dir, "SFD_GUI.py")
if os.path.exists(gui_file):
    print(f"   ✓ SFD_GUI.py found at: {gui_file}")
    # Try to read first few lines to check for syntax errors
    try:
        with open(gui_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:30]
            print(f"   ✓ File is readable (checked first 30 lines)")
    except Exception as e:
        print(f"   ✗ Error reading file: {e}")
else:
    print(f"   ✗ SFD_GUI.py not found in: {script_dir}")

print("\n" + "=" * 60)
print("Diagnostics Complete")
print("=" * 60)


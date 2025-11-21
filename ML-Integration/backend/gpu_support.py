"""
GPU Support Detection and Management
Supports NVIDIA CUDA, AMD ROCm/HIP, and falls back to CPU
Based on SFD.py GPU detection logic
"""
import platform
import logging

logger = logging.getLogger(__name__)

# Global GPU status variables
GPU_AVAILABLE = False
GPU_TYPE = None  # 'NVIDIA', 'AMD', or None
GPU_BACKEND = None  # 'CUDA', 'ROCm', 'HIP', or None
cudf = None
cp = None  # cupy (CUDA) or cupy-rocm (ROCm/HIP)
hip = None  # PyHIP for direct HIP access

is_windows = platform.system() == 'Windows'

# Try to import GPU libraries - check for NVIDIA (CUDA), AMD (ROCm/HIP), and cupy
# Note: ROCm/cupy-rocm/HIP support on Windows is limited but we'll attempt detection

# First try NVIDIA CUDA (cudf + cupy) - works on both Windows and Linux
try:
    import cudf  # type: ignore
    import cupy as cp  # type: ignore
    # Verify CUDA is actually available
    if cp.cuda.is_available():
        GPU_AVAILABLE = True
        GPU_TYPE = 'NVIDIA'
        GPU_BACKEND = 'CUDA'
    else:
        raise ImportError("CUDA not available")
except ImportError:
    # If NVIDIA libraries not available, try AMD ROCm/HIP
    # Note: ROCm/cupy-rocm/HIP support on Windows is limited but we'll attempt detection
    try:
        # Try cupy-rocm for AMD GPUs (cupy-rocm is installed as 'cupy' but uses ROCm/HIP backend)
        # First check if we can import cupy (might be cupy-rocm)
        import cupy as cp  # type: ignore
        # Check if this is ROCm version by trying to access device info
        # cupy-rocm should work similarly to cupy but for AMD GPUs using HIP
        try:
            # Try to get device info - ROCm version should work
            if hasattr(cp, 'cuda') and cp.cuda.is_available():
                GPU_AVAILABLE = True
                GPU_TYPE = 'AMD'
                GPU_BACKEND = 'ROCm'  # cupy-rocm uses ROCm which is built on HIP
            else:
                raise ImportError("ROCm not available")
        except:
            # If cupy is installed but not working, it might be regular cupy without GPU
            raise ImportError("GPU not available")
    except ImportError:
        # Try direct HIP support via PyHIP (if available)
        # Prioritize pyhip import - this is the standard PyHIP package
        try:
            # First try pyhip (the standard PyHIP package name)
            try:
                import pyhip as hip  # type: ignore
                # Check if HIP is available and working
                # PyHIP provides direct access to HIP runtime
                # Test if HIP devices are available
                try:
                    if hasattr(hip, 'is_available') and hip.is_available():
                        GPU_AVAILABLE = True
                        GPU_TYPE = 'AMD'
                        GPU_BACKEND = 'HIP'
                    elif hasattr(hip, 'getDeviceCount'):
                        # Alternative check: try to get device count
                        device_count = hip.getDeviceCount()
                        if device_count > 0:
                            GPU_AVAILABLE = True
                            GPU_TYPE = 'AMD'
                            GPU_BACKEND = 'HIP'
                        else:
                            raise ImportError("PyHIP: No HIP devices available")
                    else:
                        # If no availability check, assume it's available if imported
                        GPU_AVAILABLE = True
                        GPU_TYPE = 'AMD'
                        GPU_BACKEND = 'HIP'
                except Exception as e:
                    raise ImportError(f"PyHIP not available: {e}")
            except ImportError:
                # Try alternative HIP import name
                try:
                    import hip  # type: ignore
                    if hasattr(hip, 'is_available') and hip.is_available():
                        GPU_AVAILABLE = True
                        GPU_TYPE = 'AMD'
                        GPU_BACKEND = 'HIP'
                    elif hasattr(hip, 'getDeviceCount'):
                        device_count = hip.getDeviceCount()
                        if device_count > 0:
                            GPU_AVAILABLE = True
                            GPU_TYPE = 'AMD'
                            GPU_BACKEND = 'HIP'
                        else:
                            raise ImportError("HIP: No devices available")
                    else:
                        GPU_AVAILABLE = True
                        GPU_TYPE = 'AMD'
                        GPU_BACKEND = 'HIP'
                except ImportError:
                    raise ImportError("PyHIP libraries not found")
        except ImportError:
            # No GPU libraries available
            GPU_AVAILABLE = False
            GPU_TYPE = None
            GPU_BACKEND = None
            cudf = None
            cp = None
            hip = None

# Log GPU availability status
if GPU_AVAILABLE:
    if GPU_TYPE == 'NVIDIA':
        logger.info("GPU support detected and enabled (NVIDIA CUDA with RAPIDS libraries)")
    elif GPU_TYPE == 'AMD':
        if GPU_BACKEND == 'HIP':
            logger.info("GPU support detected and enabled (AMD HIP via PyHIP)")
            if hip is not None:
                try:
                    if hasattr(hip, 'getDeviceCount'):
                        device_count = hip.getDeviceCount()
                        logger.info(f"PyHIP: {device_count} HIP device(s) detected")
                except:
                    pass
        elif GPU_BACKEND == 'ROCm':
            logger.info("GPU support detected and enabled (AMD ROCm/HIP via cupy-rocm)")
        else:
            logger.info("GPU support detected and enabled (AMD)")
        if is_windows:
            logger.info("AMD GPU detected on Windows - functionality may be limited")
    else:
        logger.info("GPU support detected and enabled")
else:
    if is_windows:
        logger.info("GPU support not available, using CPU-based processing")
        logger.info("Note: For AMD GPUs, install cupy-rocm or PyHIP. ROCm/HIP support on Windows may be limited.")
    else:
        logger.info("GPU support not available, using CPU-based processing")


def get_gpu_info():
    """
    Get detailed GPU information
    Returns: dict with GPU status and capabilities
    """
    info = {
        'available': GPU_AVAILABLE,
        'type': GPU_TYPE,
        'backend': GPU_BACKEND,
        'platform': platform.system(),
        'device_count': 0,
        'device_names': [],
        'memory_info': {}
    }
    
    if GPU_AVAILABLE:
        try:
            if GPU_TYPE == 'NVIDIA' and cp is not None:
                # NVIDIA GPU info via cupy
                info['device_count'] = cp.cuda.runtime.getDeviceCount()
                for i in range(info['device_count']):
                    with cp.cuda.Device(i):
                        props = cp.cuda.runtime.getDeviceProperties(i)
                        name = props['name'].decode('utf-8') if isinstance(props['name'], bytes) else str(props['name'])
                        info['device_names'].append(name)
                        
                        # Get memory info
                        mem_info = cp.cuda.runtime.memGetInfo()
                        info['memory_info'][i] = {
                            'free': mem_info[0] / (1024**3),  # GB
                            'total': mem_info[1] / (1024**3)  # GB
                        }
            
            elif GPU_TYPE == 'AMD' and GPU_BACKEND == 'HIP' and hip is not None:
                # AMD GPU info via PyHIP
                try:
                    # Try multiple methods to get device count
                    device_count = 0
                    
                    if hasattr(hip, 'getDeviceCount'):
                        try:
                            device_count = hip.getDeviceCount()
                        except:
                            pass
                    
                    if device_count == 0 and hasattr(hip, 'hipGetDeviceCount'):
                        try:
                            device_count = hip.hipGetDeviceCount()
                        except:
                            pass
                    
                    # If PyHIP is available but methods fail, assume at least 1 device
                    if device_count == 0 and hip is not None:
                        device_count = 1
                        logger.info("PyHIP available but device count methods failed, assuming 1 device")
                    
                    info['device_count'] = device_count
                    
                    for i in range(device_count):
                        try:
                            if hasattr(hip, 'getDeviceProperties'):
                                props = hip.getDeviceProperties(i)
                                name = props.name if hasattr(props, 'name') else f"AMD GPU {i}"
                                info['device_names'].append(name)
                            elif hasattr(hip, 'hipGetDeviceProperties'):
                                props = hip.hipGetDeviceProperties(i)
                                name = props.name if hasattr(props, 'name') else f"AMD GPU {i}"
                                info['device_names'].append(name)
                            else:
                                info['device_names'].append(f"AMD HIP Device {i}")
                        except:
                            info['device_names'].append(f"AMD HIP Device {i}")
                except Exception as e:
                    logger.warning(f"Error getting AMD HIP device info: {e}")
            
            elif GPU_TYPE == 'AMD' and GPU_BACKEND == 'ROCm' and cp is not None:
                # AMD GPU info via cupy-rocm
                try:
                    info['device_count'] = cp.cuda.runtime.getDeviceCount()
                    for i in range(info['device_count']):
                        info['device_names'].append(f"AMD ROCm Device {i}")
                except:
                    info['device_count'] = 1
                    info['device_names'].append("AMD ROCm Device")
        
        except Exception as e:
            logger.warning(f"Error getting GPU details: {e}")
    
    return info


def convert_to_gpu_dataframe(df):
    """
    Convert pandas DataFrame to GPU DataFrame (cuDF) if GPU available
    Returns: GPU DataFrame if available, otherwise original DataFrame
    """
    if GPU_AVAILABLE and GPU_TYPE == 'NVIDIA' and cudf is not None:
        try:
            return cudf.DataFrame.from_pandas(df)
        except Exception as e:
            logger.warning(f"Failed to convert to cuDF DataFrame: {e}")
            return df
    
    # For AMD or no GPU, return original pandas DataFrame
    return df


def convert_to_cpu_dataframe(df):
    """
    Convert GPU DataFrame back to pandas DataFrame if needed
    Returns: pandas DataFrame
    """
    if GPU_AVAILABLE and GPU_TYPE == 'NVIDIA' and cudf is not None:
        try:
            if isinstance(df, cudf.DataFrame):
                return df.to_pandas()
        except Exception as e:
            logger.warning(f"Failed to convert from cuDF DataFrame: {e}")
    
    return df


def get_processing_backend():
    """
    Get the appropriate processing backend (GPU or CPU)
    Returns: tuple (use_gpu: bool, backend_name: str)
    """
    if GPU_AVAILABLE:
        return True, f"{GPU_TYPE} {GPU_BACKEND}"
    return False, "CPU"


def get_installation_instructions():
    """
    Get GPU installation instructions based on detected hardware
    Returns: dict with installation commands and notes
    """
    instructions = {
        'nvidia': {
            'title': 'NVIDIA GPU (CUDA)',
            'requirements': [
                'NVIDIA GPU with CUDA support',
                'CUDA Toolkit 11.8 or later',
                'cuDNN library'
            ],
            'install_commands': [
                'pip install cudf-cu11 dask-cudf-cu11 --extra-index-url=https://pypi.nvidia.com',
                'pip install cupy-cuda11x',
                'pip install cuml-cu11 --extra-index-url=https://pypi.nvidia.com'
            ],
            'notes': [
                'RAPIDS libraries provide fastest performance',
                'Requires compatible NVIDIA driver',
                'Works on Windows and Linux'
            ]
        },
        'amd': {
            'title': 'AMD GPU (ROCm/HIP)',
            'requirements': [
                'AMD GPU with ROCm support',
                'ROCm 5.0 or later (Linux)',
                'HIP SDK (Windows - limited support)'
            ],
            'install_commands': [
                '# Option 1: cupy-rocm',
                'pip install cupy-rocm-5-0',
                '',
                '# Option 2: PyHIP (direct HIP)',
                'pip install pyhip',
                '',
                '# For Windows (limited support):',
                '# Install AMD HIP SDK from:',
                '# https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html'
            ],
            'notes': [
                'ROCm support is primarily for Linux',
                'Windows support is experimental',
                'Performance may vary by GPU model',
                'PyHIP provides direct HIP runtime access'
            ]
        }
    }
    
    return instructions


# Export key variables and functions
__all__ = [
    'GPU_AVAILABLE',
    'GPU_TYPE',
    'GPU_BACKEND',
    'cudf',
    'cp',
    'hip',
    'get_gpu_info',
    'convert_to_gpu_dataframe',
    'convert_to_cpu_dataframe',
    'get_processing_backend',
    'get_installation_instructions'
]


"""
Comprehensive Diagnostic Tool for AIS Law Enforcement LLM
Consolidates all testing and diagnostic capabilities into one script

Usage:
    python diagnostic_tool.py              # Run all checks
    python diagnostic_tool.py --models     # Test model detection only
    python diagnostic_tool.py --claude     # Test Claude responses only
    python diagnostic_tool.py --startup    # Check startup requirements only
    python diagnostic_tool.py --gpu        # Check GPU status only
"""
import os
import sys
import argparse
from typing import Dict, Any


def check_api_key() -> tuple[bool, str]:
    """Check if API key is set"""
    if not os.environ.get('ANTHROPIC_API_KEY'):
        return False, "ANTHROPIC_API_KEY environment variable not set"
    return True, os.environ['ANTHROPIC_API_KEY']


def test_model_discovery(api_key: str) -> Dict[str, Any]:
    """Test automatic model discovery from Anthropic API"""
    print("\n" + "=" * 80)
    print("MODEL AUTO-DISCOVERY TEST")
    print("=" * 80)
    
    results = {
        "api_available": False,
        "models_found": [],
        "selected_model": None,
        "method": "unknown"
    }
    
    # Try to fetch models from API
    try:
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
            results["api_available"] = True
            models_data = response.json()
            
            if "data" in models_data:
                results["models_found"] = [m.get("id", "unknown") for m in models_data["data"]]
                results["method"] = "API Discovery"
                
                print(f"\n[OK] API Discovery successful - found {len(results['models_found'])} models:")
                for model in results["models_found"]:
                    print(f"  - {model}")
            else:
                results["method"] = "Fallback"
                print("\n[INFO] API available but no models in response, using fallback")
        else:
            results["method"] = "Fallback"
            print(f"\n[INFO] API returned {response.status_code}, using fallback model list")
    
    except Exception as e:
        results["method"] = "Fallback"
        print(f"\n[INFO] Could not query API ({str(e)}), using fallback model list")
    
    # Test actual agent creation
    try:
        from claude_agent import AISFraudDetectionAgent
        
        print("\n[TEST] Creating agent with auto-detection...")
        agent = AISFraudDetectionAgent(api_key=api_key)
        results["selected_model"] = agent.model
        
        print(f"[OK] Agent created successfully")
        print(f"[OK] Selected model: {agent.model}")
        
    except Exception as e:
        print(f"[X] Failed to create agent: {e}")
        results["error"] = str(e)
    
    return results


def test_claude_responses(api_key: str) -> Dict[str, Any]:
    """Test Claude's basic response capabilities"""
    print("\n" + "=" * 80)
    print("CLAUDE AI RESPONSE TEST")
    print("=" * 80)
    
    results = {
        "tests_passed": 0,
        "tests_failed": 0,
        "responses": []
    }
    
    try:
        from claude_agent import AISFraudDetectionAgent
        import asyncio
        
        agent = AISFraudDetectionAgent(api_key=api_key)
        print(f"\n[OK] Using model: {agent.model}\n")
        
        test_questions = [
            "What are you?",
            "What can you do?",
            "What is AIS?",
        ]
        
        async def run_tests():
            for i, question in enumerate(test_questions, 1):
                print(f"\n{'=' * 80}")
                print(f"TEST {i}/{len(test_questions)}: {question}")
                print('=' * 80)
                
                try:
                    response = await agent.chat(question, session_context={})
                    message = response.get('message', 'No response')
                    
                    # Handle Unicode for Windows
                    try:
                        print(f"\nResponse preview (first 200 chars):")
                        print("-" * 80)
                        print(message[:200] + "..." if len(message) > 200 else message)
                    except UnicodeEncodeError:
                        print(message.encode('ascii', errors='replace').decode('ascii')[:200] + "...")
                    
                    print(f"\n[OK] Test {i} passed")
                    results["tests_passed"] += 1
                    results["responses"].append({"question": question, "success": True})
                    
                except Exception as e:
                    print(f"[X] Test {i} failed: {e}")
                    results["tests_failed"] += 1
                    results["responses"].append({"question": question, "success": False, "error": str(e)})
        
        asyncio.run(run_tests())
        
    except Exception as e:
        print(f"[X] Failed to run tests: {e}")
        results["error"] = str(e)
    
    return results


def check_gpu_status() -> Dict[str, Any]:
    """Check GPU availability and status"""
    print("\n" + "=" * 80)
    print("GPU STATUS CHECK")
    print("=" * 80)
    
    results = {
        "gpu_available": False,
        "gpu_type": None,
        "gpu_backend": None,
        "details": []
    }
    
    try:
        from gpu_support import (
            GPU_AVAILABLE, GPU_TYPE, GPU_BACKEND,
            get_gpu_info, get_installation_instructions
        )
        
        results["gpu_available"] = GPU_AVAILABLE
        results["gpu_type"] = GPU_TYPE
        results["gpu_backend"] = GPU_BACKEND
        
        print(f"\n[INFO] GPU Available: {GPU_AVAILABLE}")
        
        if GPU_AVAILABLE:
            print(f"[OK] GPU Type: {GPU_TYPE}")
            print(f"[OK] GPU Backend: {GPU_BACKEND}")
            
            gpu_info = get_gpu_info()
            print(f"\n{gpu_info}")
            results["details"].append(gpu_info)
        else:
            print(f"[INFO] Running in CPU-only mode")
            instructions = get_installation_instructions()
            print(f"\n{instructions}")
            results["details"].append(instructions)
    
    except Exception as e:
        print(f"[X] Error checking GPU: {e}")
        results["error"] = str(e)
    
    return results


def check_startup_requirements() -> Dict[str, Any]:
    """Check all startup requirements"""
    print("\n" + "=" * 80)
    print("STARTUP REQUIREMENTS CHECK")
    print("=" * 80)
    
    results = {
        "dependencies": [],
        "missing_dependencies": [],
        "all_ok": True
    }
    
    required_modules = [
        "fastapi",
        "uvicorn",
        "anthropic",
        "pandas",
        "numpy",
        "boto3",
        "shapely",
        "pyproj",
        "matplotlib",
        "plotly",
        "folium",
        "openpyxl",
        "requests",
    ]
    
    print("\n[TEST] Checking required dependencies...")
    print("-" * 80)
    
    for module_name in required_modules:
        try:
            __import__(module_name)
            print(f"[OK] {module_name}")
            results["dependencies"].append(module_name)
        except ImportError:
            print(f"[X] {module_name} - MISSING")
            results["missing_dependencies"].append(module_name)
            results["all_ok"] = False
    
    if results["all_ok"]:
        print(f"\n[OK] All {len(required_modules)} required dependencies installed")
    else:
        print(f"\n[X] Missing {len(results['missing_dependencies'])} dependencies")
        print(f"\nTo install missing dependencies:")
        print(f"  pip install {' '.join(results['missing_dependencies'])}")
    
    return results


def print_summary(all_results: Dict[str, Any]):
    """Print summary of all diagnostic results"""
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    
    # API Key
    if all_results.get("api_key_set"):
        print("\n[OK] API Key: Set")
    else:
        print("\n[X] API Key: NOT SET")
    
    # Dependencies
    if "startup" in all_results:
        startup = all_results["startup"]
        if startup.get("all_ok"):
            print(f"[OK] Dependencies: All {len(startup['dependencies'])} installed")
        else:
            print(f"[X] Dependencies: {len(startup['missing_dependencies'])} missing")
    
    # GPU
    if "gpu" in all_results:
        gpu = all_results["gpu"]
        if gpu.get("gpu_available"):
            print(f"[OK] GPU: Available ({gpu['gpu_type']} - {gpu['gpu_backend']})")
        else:
            print(f"[INFO] GPU: Not available (CPU mode)")
    
    # Model Discovery
    if "models" in all_results:
        models = all_results["models"]
        if models.get("selected_model"):
            print(f"[OK] Model: {models['selected_model']}")
            print(f"     Method: {models['method']}")
        else:
            print(f"[X] Model: Detection failed")
    
    # Claude Tests
    if "claude" in all_results:
        claude = all_results["claude"]
        passed = claude.get("tests_passed", 0)
        failed = claude.get("tests_failed", 0)
        if failed == 0 and passed > 0:
            print(f"[OK] Claude Tests: {passed}/{passed + failed} passed")
        else:
            print(f"[INFO] Claude Tests: {passed}/{passed + failed} passed")
    
    print("\n" + "=" * 80)
    print("READY TO START")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Start backend: python app.py")
    print("  2. Open browser: frontend/setup.html")
    print("  3. Configure data source and start investigating!")
    print()


def main():
    """Main diagnostic function"""
    parser = argparse.ArgumentParser(description="AIS Law Enforcement LLM Diagnostic Tool")
    parser.add_argument("--models", action="store_true", help="Test model detection only")
    parser.add_argument("--claude", action="store_true", help="Test Claude responses only")
    parser.add_argument("--startup", action="store_true", help="Check startup requirements only")
    parser.add_argument("--gpu", action="store_true", help="Check GPU status only")
    
    args = parser.parse_args()
    
    # If no specific test selected, run all
    run_all = not (args.models or args.claude or args.startup or args.gpu)
    
    print("=" * 80)
    print("AIS LAW ENFORCEMENT LLM - DIAGNOSTIC TOOL")
    print("=" * 80)
    
    all_results = {}
    
    # Check API key first (needed for most tests)
    api_key_ok, api_key = check_api_key()
    all_results["api_key_set"] = api_key_ok
    
    if not api_key_ok and (args.models or args.claude or run_all):
        print("\n" + "=" * 80)
        print("[X] ERROR: ANTHROPIC_API_KEY not set")
        print("=" * 80)
        print("\nPlease set your Claude API key:")
        print("Windows PowerShell:")
        print('  $env:ANTHROPIC_API_KEY="your-key-here"')
        print("\nWindows CMD:")
        print('  set ANTHROPIC_API_KEY=your-key-here')
        print("\nLinux/Mac:")
        print('  export ANTHROPIC_API_KEY=your-key-here')
        print("=" * 80)
        
        # Can still run startup and GPU checks without API key
        if not (args.startup or args.gpu):
            sys.exit(1)
    
    # Run requested diagnostics
    if args.startup or run_all:
        all_results["startup"] = check_startup_requirements()
    
    if args.gpu or run_all:
        all_results["gpu"] = check_gpu_status()
    
    if (args.models or run_all) and api_key_ok:
        all_results["models"] = test_model_discovery(api_key)
    
    if (args.claude or run_all) and api_key_ok:
        all_results["claude"] = test_claude_responses(api_key)
    
    # Print summary
    if run_all:
        print_summary(all_results)


if __name__ == "__main__":
    main()


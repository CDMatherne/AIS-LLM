"""
Benchmark tool for comparing the original and optimized Claude agents

This tool runs a series of test queries against both the original and optimized
Claude agents and reports on response times, token usage, and costs.
"""
import os
import time
import asyncio
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Import the agents
from backend.claude_agent import AISFraudDetectionAgent
from backend.optimized_claude_agent import OptimizedAISFraudDetectionAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sample queries of varying complexity
SAMPLE_QUERIES = [
    # Simple queries (should route to Haiku)
    "What is AIS data?",
    "List the vessel types you can detect.",
    "What time period of data is available?",
    
    # Medium complexity queries (might route to Haiku or Sonnet)
    "Explain how AIS beacon off anomalies work.",
    "What are the key maritime fraud patterns to watch for?",
    "How do you detect vessel rendezvous events?",
    
    # Complex queries (should route to Sonnet)
    "Analyze the common patterns in illegal fishing operations and provide a detailed breakdown of detection strategies.",
    "Compare and contrast the effectiveness of different anomaly detection approaches for maritime smuggling operations.",
    "Develop a comprehensive investigation strategy for vessels suspected of sanctions evasion with multiple data points.",
]

async def benchmark_agent(agent_type: str, api_key: str, queries: List[str]) -> Dict[str, Any]:
    """
    Benchmark an agent's performance on a set of queries
    
    Args:
        agent_type: "original" or "optimized"
        api_key: Anthropic API key
        queries: List of queries to test
        
    Returns:
        Dictionary with benchmark results
    """
    results = {
        "agent_type": agent_type,
        "total_time": 0,
        "queries": [],
        "errors": 0,
    }
    
    # Create the appropriate agent
    if agent_type == "original":
        agent = AISFraudDetectionAgent(api_key=api_key)
    elif agent_type == "optimized":
        agent = OptimizedAISFraudDetectionAgent(api_key=api_key)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
        
    # Process each query
    for i, query in enumerate(queries):
        query_result = {
            "query": query,
            "query_number": i + 1,
            "time_seconds": 0,
            "success": False,
            "error": None
        }
        
        try:
            logger.info(f"[{agent_type}] Processing query {i+1}/{len(queries)}: {query[:50]}...")
            start_time = time.time()
            
            # Process the query
            response = await agent.chat(query, {})
            
            # Record the results
            elapsed = time.time() - start_time
            query_result["time_seconds"] = elapsed
            query_result["success"] = True
            query_result["response_length"] = len(response["message"]) if "message" in response else 0
            
            # If the agent used the optimizer, get the model used
            if hasattr(agent, "optimizer") and hasattr(agent.optimizer, "performance_stats"):
                query_result["model_used"] = list(agent.optimizer.performance_stats["model_usage"].keys())[-1]
                
            logger.info(f"[{agent_type}] Query {i+1} completed in {elapsed:.2f}s")
            
        except Exception as e:
            elapsed = time.time() - start_time
            query_result["time_seconds"] = elapsed
            query_result["success"] = False
            query_result["error"] = str(e)
            results["errors"] += 1
            
            logger.error(f"[{agent_type}] Query {i+1} failed: {str(e)}")
            
        finally:
            results["total_time"] += query_result["time_seconds"]
            results["queries"].append(query_result)
            
    # Calculate summary statistics
    results["average_time"] = results["total_time"] / len(queries)
    results["success_rate"] = (len(queries) - results["errors"]) / len(queries)
    
    # Get performance stats if available
    if agent_type == "optimized" and hasattr(agent, "get_performance_stats"):
        results["performance_stats"] = agent.get_performance_stats()
        
    return results

async def run_benchmark(api_key: str, queries: List[str] = None) -> Dict[str, Any]:
    """
    Run benchmark on both original and optimized agents
    
    Args:
        api_key: Anthropic API key
        queries: Optional list of test queries (uses defaults if None)
        
    Returns:
        Dictionary with benchmark results
    """
    if not queries:
        queries = SAMPLE_QUERIES
        
    logger.info(f"Starting benchmark with {len(queries)} queries")
    
    # Run the benchmarks
    optimized_results = await benchmark_agent("optimized", api_key, queries)
    original_results = await benchmark_agent("original", api_key, queries)
    
    # Calculate improvements
    if original_results["average_time"] > 0:
        speed_improvement = original_results["average_time"] / optimized_results["average_time"]
    else:
        speed_improvement = 0
        
    benchmark_results = {
        "timestamp": datetime.now().isoformat(),
        "num_queries": len(queries),
        "original": original_results,
        "optimized": optimized_results,
        "speed_improvement": speed_improvement,
        "speed_improvement_percent": (speed_improvement - 1) * 100
    }
    
    return benchmark_results
    
def print_benchmark_results(results: Dict[str, Any]):
    """
    Print formatted benchmark results
    
    Args:
        results: Benchmark results dictionary
    """
    print("\n" + "="*80)
    print(f"CLAUDE AGENT BENCHMARK RESULTS - {results['timestamp']}")
    print("="*80)
    
    print(f"\nTested {results['num_queries']} queries on both original and optimized agents")
    
    print("\nPERFORMANCE SUMMARY:")
    print(f"  Original Agent:   {results['original']['average_time']:.2f}s avg response time")
    print(f"  Optimized Agent:  {results['optimized']['average_time']:.2f}s avg response time")
    print(f"  Speed Improvement: {results['speed_improvement']:.2f}x faster ({results['speed_improvement_percent']:.1f}%)")
    
    if "performance_stats" in results["optimized"]:
        stats = results["optimized"]["performance_stats"]
        print("\nOPTIMIZED AGENT STATISTICS:")
        print(f"  Cache Hit Rate:   {stats.get('cache_hit_rate', 0):.2%}")
        print(f"  Models Used:      {', '.join(stats.get('model_usage', {}).keys())}")
        print(f"  Token Usage:      {stats.get('total_input_tokens', 0)} input, {stats.get('total_output_tokens', 0)} output")
    
    print("\nQUERY BREAKDOWN:")
    for i, (orig, opt) in enumerate(zip(results["original"]["queries"], results["optimized"]["queries"])):
        print(f"\n  Query {i+1}: \"{orig['query'][:50]}...\"")
        print(f"    Original:  {orig['time_seconds']:.2f}s {'✓' if orig['success'] else '✗'}")
        print(f"    Optimized: {opt['time_seconds']:.2f}s {'✓' if opt['success'] else '✗'} {getattr(opt, 'model_used', '')}")
        
    print("\n" + "="*80)
    
async def main():
    """Main function for the benchmark tool"""
    parser = argparse.ArgumentParser(description="Benchmark Claude Agents")
    parser.add_argument("--api-key", help="Anthropic API Key (defaults to ANTHROPIC_API_KEY env var)")
    args = parser.parse_args()
    
    # Get API key from args or environment
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("No API key provided. Use --api-key or set ANTHROPIC_API_KEY environment variable.")
        
    # Run the benchmark
    results = await run_benchmark(api_key)
    
    # Print the results
    print_benchmark_results(results)
    
if __name__ == "__main__":
    asyncio.run(main())

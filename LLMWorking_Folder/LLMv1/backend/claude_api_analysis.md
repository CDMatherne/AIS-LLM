# Claude API Implementation Analysis

## Problem Statement
The Claude API call "doesn't seem to work" - need to investigate the implementation.

## Key Areas to Check:

### 1. Agent Initialization
The app uses `OptimizedAISFraudDetectionAgent` which is imported from `optimized_claude_agent` module.

**Location in app.py:**
- Line 80: `OptimizedAISFraudDetectionAgent = optimized_claude_agent.OptimizedAISFraudDetectionAgent`
- Line 412: `agent = OptimizedAISFraudDetectionAgent(config['claude_api_key'])`
- Line 495: `agent = OptimizedAISFraudDetectionAgent(claude_api_key)`

### 2. Agent Usage
The agent is called via:
```python
response = await agent.chat(
    message.message,
    session_context={"map_context": message.map_context}
)
```

### 3. Potential Issues

#### Issue A: Module Not Found
The app imports: `optimized_claude_agent = import_local_module('optimized_claude_agent')`

This module needs to exist and be properly implemented. Without seeing the module, possible issues:
- Module doesn't exist
- Module exists but has errors
- Module's `chat()` method is not properly async
- Module doesn't properly call Claude API

#### Issue B: API Key Issues
The agent is initialized with:
```python
agent = OptimizedAISFraudDetectionAgent(config['claude_api_key'])
```

Possible issues:
- API key not set in environment
- API key invalid
- API key not passed to the Anthropic client properly

#### Issue C: Missing Anthropic Client
Looking at the app.py initialization (lines 233-260), there's:
```python
if os.getenv('CLAUDE_API_KEY'):
    anthropic_client = anthropic.Anthropic(api_key=os.getenv('CLAUDE_API_KEY'))

resilient_bedrock_client = get_resilient_client(
    bedrock_client=bedrock_client,
    anthropic_client=anthropic_client,
    ...
)
```

But this client is only used for bedrock resilience, NOT for the agent itself.

#### Issue D: Agent Implementation Missing
The `OptimizedAISFraudDetectionAgent` class needs to:
1. Accept the API key in __init__
2. Create an Anthropic client
3. Implement an async chat() method
4. Handle tool calls properly
5. Return responses in the expected format

## What the Agent Should Look Like

```python
import anthropic
from typing import Dict, Any, List

class OptimizedAISFraudDetectionAgent:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-20250514"  # or another model
        self.conversation_history = []
        
    async def chat(self, message: str, session_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Send a message to Claude and get response
        
        Returns:
            Dict with 'message' key and optional 'tool_calls' key
        """
        # Build the messages array with conversation history
        messages = self.conversation_history + [
            {"role": "user", "content": message}
        ]
        
        # Call Claude API
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=messages,
            tools=self._get_tools()  # Define tools
        )
        
        # Store in conversation history
        self.conversation_history.append({"role": "user", "content": message})
        
        # Process response
        if response.stop_reason == "tool_use":
            # Extract tool calls
            tool_calls = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "name": block.name,
                        "input": block.input
                    })
            
            # Store assistant message in history
            self.conversation_history.append({
                "role": "assistant",
                "content": response.content
            })
            
            return {
                "message": "",  # No text message when tool use
                "tool_calls": tool_calls
            }
        else:
            # Extract text response
            text_content = ""
            for block in response.content:
                if block.type == "text":
                    text_content += block.text
            
            # Store in history
            self.conversation_history.append({
                "role": "assistant",
                "content": text_content
            })
            
            return {
                "message": text_content
            }
    
    async def process_tool_results(self, tool_results: List[Dict]) -> Dict[str, Any]:
        """
        Process tool results and get Claude's next response
        """
        # Add tool results to conversation
        tool_result_content = []
        for result in tool_results:
            tool_result_content.append({
                "type": "tool_result",
                "tool_use_id": result["tool_call_id"],
                "content": str(result["result"])
            })
        
        self.conversation_history.append({
            "role": "user",
            "content": tool_result_content
        })
        
        # Get Claude's response
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=self.conversation_history,
            tools=self._get_tools()
        )
        
        # Process similar to chat()
        # ... (implementation continues)
        
    def _get_tools(self) -> List[Dict]:
        """Define available tools for Claude"""
        # Return tool definitions
        return [...]
    
    def get_conversation_length(self) -> int:
        """Get length of conversation history"""
        return len(self.conversation_history)
```

## Common Problems and Solutions

### Problem 1: "No module named 'optimized_claude_agent'"
**Solution:** The module doesn't exist or isn't in the Python path.
- Check if the file exists: `optimized_claude_agent.py` or `backend/optimized_claude_agent.py`
- Verify the import_local_module() helper is working

### Problem 2: "Agent has no attribute 'chat'"
**Solution:** The OptimizedAISFraudDetectionAgent class is missing the chat method.
- Implement the async chat() method
- Ensure it returns a dict with 'message' key

### Problem 3: "Anthropic API error: 401 Unauthorized"
**Solution:** API key issue.
- Check CLAUDE_API_KEY environment variable is set
- Verify API key is valid
- Ensure API key is passed to Anthropic client

### Problem 4: "coroutine was never awaited"
**Solution:** Async/await issue.
- Ensure chat() method is defined as `async def`
- Ensure it's called with `await agent.chat(...)`

### Problem 5: Tool calls not working
**Solution:** Tool definitions or processing issue.
- Verify tools are properly defined in _get_tools()
- Check tool response parsing
- Ensure process_tool_results() is implemented

## Diagnostic Steps

1. **Check if module exists:**
   ```bash
   find . -name "optimized_claude_agent.py"
   ```

2. **Check API key:**
   ```bash
   echo $CLAUDE_API_KEY
   ```

3. **Test agent initialization:**
   ```python
   from optimized_claude_agent import OptimizedAISFraudDetectionAgent
   agent = OptimizedAISFraudDetectionAgent("test_key")
   print(dir(agent))  # Should show 'chat' method
   ```

4. **Test API call:**
   ```python
   import anthropic
   client = anthropic.Anthropic(api_key="your_key")
   response = client.messages.create(
       model="claude-sonnet-4-20250514",
       max_tokens=1024,
       messages=[{"role": "user", "content": "Hello!"}]
   )
   print(response)
   ```

## Recommended Fix

If the optimized_claude_agent module is missing or broken, you need to:

1. **Create the module** with proper Claude API integration
2. **Implement async chat()** method
3. **Handle tool calls** properly
4. **Manage conversation history**
5. **Handle errors** gracefully

Would you like me to create a complete implementation of the OptimizedAISFraudDetectionAgent class?

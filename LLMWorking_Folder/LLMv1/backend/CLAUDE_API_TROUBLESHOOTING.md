# Claude API Implementation Issue - Troubleshooting Guide

## Problem: "Claude API call doesn't seem to work"

---

## 🔍 DIAGNOSIS CHECKLIST

### Check 1: Does the optimized_claude_agent module exist?

**Test:**
```bash
find . -name "optimized_claude_agent.py" -o -name "optimized_claude_agent"
```

**Possible Results:**
- ✅ File exists → Go to Check 2
- ❌ File missing → **ROOT CAUSE: Module doesn't exist**

**Fix if missing:**
Use the reference implementation provided in `optimized_claude_agent_reference.py`

---

### Check 2: Is the CLAUDE_API_KEY set?

**Test:**
```bash
echo $CLAUDE_API_KEY
# or in Python:
import os
print(os.getenv('CLAUDE_API_KEY'))
```

**Possible Results:**
- ✅ Key is set and looks valid (starts with `sk-ant-...`) → Go to Check 3
- ❌ Key is None or empty → **ROOT CAUSE: API key not configured**
- ❌ Key looks invalid → **ROOT CAUSE: Invalid API key**

**Fix:**
```bash
export CLAUDE_API_KEY="sk-ant-api03-your-key-here"
```

---

### Check 3: Is the agent class properly initialized?

**Test:**
```python
from optimized_claude_agent import OptimizedAISFraudDetectionAgent
agent = OptimizedAISFraudDetectionAgent("test-key")
print(type(agent))
print(hasattr(agent, 'chat'))
print(hasattr(agent, 'client'))
```

**Possible Results:**
- ✅ All checks pass → Go to Check 4
- ❌ ImportError → Module has syntax errors or missing dependencies
- ❌ `chat` method missing → Implementation incomplete
- ❌ `client` attribute missing → Anthropic client not initialized

**Common Issues:**
```python
# ❌ WRONG - No Anthropic client
class OptimizedAISFraudDetectionAgent:
    def __init__(self, api_key: str):
        self.api_key = api_key  # Stored but not used!
        # Missing: self.client = anthropic.Anthropic(api_key=api_key)

# ✅ CORRECT
class OptimizedAISFraudDetectionAgent:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.conversation_history = []
```

---

### Check 4: Is the chat method async and properly implemented?

**Test:**
```python
import inspect
from optimized_claude_agent import OptimizedAISFraudDetectionAgent

agent = OptimizedAISFraudDetectionAgent("test-key")
print(f"chat is async: {inspect.iscoroutinefunction(agent.chat)}")
print(f"chat signature: {inspect.signature(agent.chat)}")
```

**Possible Results:**
- ✅ `chat is async: True` → Go to Check 5
- ❌ `chat is async: False` → **ROOT CAUSE: Method not async**

**Common Issue:**
```python
# ❌ WRONG - Not async
def chat(self, message: str) -> Dict[str, Any]:
    response = self.client.messages.create(...)  # Won't work in async context
    return {"message": response.text}

# ✅ CORRECT
async def chat(self, message: str) -> Dict[str, Any]:
    response = self.client.messages.create(...)
    return {"message": response.content[0].text}
```

---

### Check 5: Can we make a test API call?

**Test:**
```python
import anthropic
import os

api_key = os.getenv('CLAUDE_API_KEY')
client = anthropic.Anthropic(api_key=api_key)

try:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Say hello"}]
    )
    print("✅ API call successful!")
    print(f"Response: {response.content[0].text}")
except anthropic.APIError as e:
    print(f"❌ API Error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
```

**Possible Results:**
- ✅ Success → API key and connection work, issue is in agent implementation
- ❌ 401 Unauthorized → Invalid API key
- ❌ 429 Rate limit → API quota exceeded
- ❌ Connection error → Network issue

---

## 🐛 COMMON BUGS AND FIXES

### Bug 1: Response Processing Error

**Symptom:**
```
AttributeError: 'Message' object has no attribute 'text'
```

**Cause:**
Trying to access `response.text` when it's actually `response.content[0].text`

**Fix:**
```python
# ❌ WRONG
response = client.messages.create(...)
text = response.text  # Doesn't exist!

# ✅ CORRECT
response = client.messages.create(...)
text = ""
for block in response.content:
    if block.type == "text":
        text += block.text
```

---

### Bug 2: Tool Calls Not Detected

**Symptom:**
Agent never executes tools, always returns text only

**Cause:**
- Tools not defined in API call
- stop_reason check missing
- Tool extraction logic missing

**Fix:**
```python
# ✅ CORRECT - Include tools in API call
response = self.client.messages.create(
    model=self.model,
    max_tokens=4096,
    messages=messages,
    tools=self._get_tools()  # ← Must include this!
)

# ✅ CORRECT - Check for tool use
if response.stop_reason == "tool_use":
    tool_calls = []
    for block in response.content:
        if block.type == "tool_use":
            tool_calls.append({
                "id": block.id,
                "name": block.name,
                "input": block.input
            })
    return {"message": "", "tool_calls": tool_calls}
```

---

### Bug 3: Conversation History Not Maintained

**Symptom:**
Claude forgets previous messages, doesn't maintain context

**Cause:**
Not appending messages to conversation history

**Fix:**
```python
# ✅ CORRECT - Maintain history
async def chat(self, message: str) -> Dict[str, Any]:
    # Create user message
    user_message = {"role": "user", "content": message}
    
    # Add to history BEFORE sending
    messages = self.conversation_history + [user_message]
    
    # Call API
    response = self.client.messages.create(
        model=self.model,
        max_tokens=4096,
        messages=messages  # Use full history
    )
    
    # Store in history AFTER response
    self.conversation_history.append(user_message)
    self.conversation_history.append({
        "role": "assistant",
        "content": response.content
    })
    
    return {"message": text_content}
```

---

### Bug 4: Tool Results Not Sent Back to Claude

**Symptom:**
Tool executes but Claude doesn't respond with interpretation

**Cause:**
Missing `process_tool_results()` method or incorrect implementation

**Fix:**
```python
# ✅ CORRECT
async def process_tool_results(self, tool_results: List[Dict]) -> Dict[str, Any]:
    # Build tool result content
    tool_result_content = []
    for result in tool_results:
        tool_result_content.append({
            "type": "tool_result",
            "tool_use_id": result["tool_call_id"],
            "content": json.dumps(result["result"])
        })
    
    # Add to conversation
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
    
    # Process and return
    return self._process_response(response)
```

---

### Bug 5: System Prompt Not Applied

**Symptom:**
Claude doesn't follow instructions or context properly

**Cause:**
System prompt not included in API call

**Fix:**
```python
# ✅ CORRECT
response = self.client.messages.create(
    model=self.model,
    max_tokens=4096,
    system=self.system_prompt,  # ← Must include this!
    messages=messages,
    tools=self._get_tools()
)
```

---

## 🔧 STEP-BY-STEP FIX PROCESS

### Step 1: Verify Module Exists

```bash
ls -la optimized_claude_agent.py
# or
ls -la backend/optimized_claude_agent.py
```

If missing, copy the reference implementation:
```bash
cp optimized_claude_agent_reference.py optimized_claude_agent.py
```

---

### Step 2: Test Basic Import

```python
try:
    from optimized_claude_agent import OptimizedAISFraudDetectionAgent
    print("✅ Import successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
```

---

### Step 3: Test Agent Initialization

```python
import os
api_key = os.getenv('CLAUDE_API_KEY', 'test-key')

try:
    agent = OptimizedAISFraudDetectionAgent(api_key)
    print("✅ Agent initialized")
    print(f"Has chat method: {hasattr(agent, 'chat')}")
    print(f"Has client: {hasattr(agent, 'client')}")
except Exception as e:
    print(f"❌ Initialization failed: {e}")
```

---

### Step 4: Test Simple Chat

```python
import asyncio
import os

async def test():
    api_key = os.getenv('CLAUDE_API_KEY')
    if not api_key:
        print("❌ No API key set")
        return
    
    agent = OptimizedAISFraudDetectionAgent(api_key)
    
    try:
        response = await agent.chat("Hello! Say hi back.")
        print(f"✅ Chat successful: {response['message']}")
    except Exception as e:
        print(f"❌ Chat failed: {e}")

asyncio.run(test())
```

---

### Step 5: Test Tool Use

```python
async def test_tools():
    api_key = os.getenv('CLAUDE_API_KEY')
    agent = OptimizedAISFraudDetectionAgent(api_key)
    
    try:
        response = await agent.chat("Please set the analysis timespan to October 15-17, 2024")
        
        if response.get('tool_calls'):
            print(f"✅ Tool calls detected: {len(response['tool_calls'])}")
            for tc in response['tool_calls']:
                print(f"  - {tc['name']}: {tc['input']}")
        else:
            print("❌ No tool calls detected")
            print(f"Response: {response['message']}")
    except Exception as e:
        print(f"❌ Test failed: {e}")

asyncio.run(test_tools())
```

---

## 📋 QUICK CHECKLIST

- [ ] optimized_claude_agent.py file exists
- [ ] CLAUDE_API_KEY environment variable is set
- [ ] Agent class has `__init__` that creates anthropic.Anthropic client
- [ ] Agent has `async def chat()` method
- [ ] Agent has `async def process_tool_results()` method
- [ ] chat() method includes tools in API call
- [ ] chat() method checks for `stop_reason == "tool_use"`
- [ ] chat() method extracts tool calls from response.content
- [ ] Conversation history is maintained
- [ ] System prompt is included in API calls
- [ ] Tool definitions match available tools in app.py

---

## 🚀 RECOMMENDED SOLUTION

If you're having persistent issues, I recommend:

1. **Replace the existing optimized_claude_agent.py** with the reference implementation provided
2. **Verify all dependencies are installed:**
   ```bash
   pip install anthropic
   ```
3. **Set the API key:**
   ```bash
   export CLAUDE_API_KEY="your-key-here"
   ```
4. **Test the agent in isolation** before integrating with app.py
5. **Check the logs** when running the app for specific error messages

---

## 📞 ERROR MESSAGE DECODER

| Error Message | Likely Cause | Fix |
|--------------|--------------|-----|
| `No module named 'optimized_claude_agent'` | File doesn't exist | Copy reference implementation |
| `'NoneType' object has no attribute 'chat'` | Agent initialization failed | Check API key, check __init__ |
| `coroutine was never awaited` | Missing await | Add `await` before agent.chat() |
| `401 Unauthorized` | Invalid API key | Check CLAUDE_API_KEY value |
| `'Message' object has no attribute 'text'` | Wrong response parsing | Use response.content[0].text |
| `list index out of range` | Empty response.content | Check API call parameters |
| `Tool 'X' not found` | Tool definition mismatch | Verify tool names match |

---

## 🎯 FINAL VERIFICATION

Once fixed, test the complete flow:

```python
import asyncio
import os

async def full_test():
    from optimized_claude_agent import OptimizedAISFraudDetectionAgent
    
    api_key = os.getenv('CLAUDE_API_KEY')
    agent = OptimizedAISFraudDetectionAgent(api_key)
    
    # Test 1: Simple chat
    response = await agent.chat("Hello!")
    assert 'message' in response
    print("✅ Test 1: Simple chat works")
    
    # Test 2: Tool use
    response = await agent.chat("Set timespan to Oct 15-17, 2024")
    assert 'tool_calls' in response or 'message' in response
    print("✅ Test 2: Tool use detection works")
    
    # Test 3: Conversation continuity
    response = await agent.chat("What did I just ask you?")
    assert 'message' in response
    print("✅ Test 3: Conversation history works")
    
    print("\n🎉 All tests passed! Agent is working correctly.")

asyncio.run(full_test())
```

---

Would you like me to:
1. Create a complete working implementation of the agent?
2. Help debug specific error messages you're seeing?
3. Provide more detailed examples of any particular issue?

# Environment Configuration Guide

## Overview
This guide explains how to configure the `.env` file for console debugging and application configuration.

## Quick Start

1. **Copy the template**: The `.env` file is located in the `backend/` directory
2. **Fill in your values**: Update the placeholder values with your actual credentials
3. **Set debug level**: Set `LOG_LEVEL=DEBUG` for detailed console output
4. **Restart the application**: The app will automatically load the `.env` file on startup

## Console Debugging Configuration

### Enable Debug Logging

To enable detailed console debugging, set these values in your `.env` file:

```env
LOG_LEVEL=DEBUG
ENABLE_CONSOLE_LOGGING=true
LOG_FORMAT=detailed
```

### Log Levels

- **DEBUG**: Shows all messages including detailed debugging information
- **INFO**: Shows informational messages (default)
- **WARNING**: Shows warnings and errors only
- **ERROR**: Shows errors only
- **CRITICAL**: Shows critical errors only

### Log Formats

- **detailed**: Full format with filename and line numbers
  ```
  2024-01-15 10:30:45 - app - INFO - [app.py:123] - Message here
  ```

- **simple**: Basic format
  ```
  2024-01-15 10:30:45 - INFO - Message here
  ```

## Required Configuration

### Claude API Key (Required)
```env
CLAUDE_API_KEY=sk-ant-api03-your-actual-key-here
```

Get your API key from: https://console.anthropic.com/

### Data Source Configuration

**For Local Data:**
```env
DATA_SOURCE=local
LOCAL_DATA_PATH=./data
LOCAL_FILE_FORMAT=auto
```

**For AWS S3:**
```env
DATA_SOURCE=aws
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-east-1
AWS_BUCKET=your-bucket-name
AWS_PREFIX=optional/path/prefix
```

## Debugging Flags

Enable additional debugging features:

```env
# Verbose error messages
VERBOSE_ERRORS=true

# Log all API requests/responses
LOG_REQUESTS=true

# Log all WebSocket messages
LOG_WEBSOCKET=true

# Log all tool executions
LOG_TOOL_EXECUTION=true
```

## Example .env File for Debugging

```env
# Console Debugging Configuration
LOG_LEVEL=DEBUG
ENABLE_CONSOLE_LOGGING=true
LOG_FORMAT=detailed
ENABLE_FILE_LOGGING=true

# Required API Key
CLAUDE_API_KEY=your-claude-api-key-here

# Data Source (local)
DATA_SOURCE=local
LOCAL_DATA_PATH=./data

# Output Directory
OUTPUT_DIRECTORY=./output

# Debugging Flags
VERBOSE_ERRORS=true
LOG_REQUESTS=true
LOG_WEBSOCKET=true
LOG_TOOL_EXECUTION=true
```

## File Locations

The application looks for `.env` files in this order:
1. Project root directory (`LLMv1/.env`)
2. Backend directory (`LLMv1/backend/.env`)

## Security Notes

⚠️ **IMPORTANT**: 
- Never commit `.env` files with real credentials to version control
- Add `.env` to `.gitignore` to prevent accidental commits
- Use environment variables or secure vaults in production
- Keep your API keys secure and rotate them regularly

## Troubleshooting

### Logs Not Appearing in Console

1. Check `ENABLE_CONSOLE_LOGGING=true` is set
2. Verify `LOG_LEVEL=DEBUG` for detailed output
3. Restart the application after changing `.env` file

### Environment Variables Not Loading

1. Ensure `.env` file is in the correct location
2. Check file permissions (should be readable)
3. Verify no syntax errors in `.env` file (no spaces around `=`)
4. Restart the application

### Too Much Log Output

If you're getting too much debug output:
- Set `LOG_LEVEL=INFO` or `LOG_LEVEL=WARNING`
- Set `LOG_REQUESTS=false` to disable request logging
- Set `LOG_WEBSOCKET=false` to disable WebSocket logging

## Testing the Configuration

After setting up your `.env` file, start the application:

```bash
# From the backend directory
python app.py

# Or using uvicorn
uvicorn app:app --host 0.0.0.0 --port 8000
```

You should see console output like:
```
2024-01-15 10:30:45 - app - INFO - [app.py:57] - Logging configured: Level=DEBUG, Console=enabled
2024-01-15 10:30:45 - config_manager - INFO - Loaded environment variables from /path/to/.env
```

## Additional Resources

- See `config_manager.py` for how configuration is loaded
- Check `interaction_logger.py` for file-based logging configuration
- Review application logs in `backend/logs/` directory


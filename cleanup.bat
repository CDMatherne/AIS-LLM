@echo off
echo Cleaning up temporary files from consolidation...

echo Removing temporary files...
del /q "app.py.new"
del /q "backend\direct_bedrock_patch_new.py"
del /q "backend\bedrock_resilience_merged.py"
del /q "README.md.new"

echo Removing unnecessary files...
del /q "backend\aws_bedrock_resilience.py"
del /q "LLM_Optimization_Plan_Summary.md"

echo Copying updated files...
copy /y "backend\bedrock_resilience_simplified.py" "backend\bedrock_resilience.py"

echo Cleanup complete!
pause

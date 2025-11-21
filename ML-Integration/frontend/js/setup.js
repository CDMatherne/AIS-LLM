/**
 * AIS Law Enforcement Assistant - Setup Page JavaScript
 */

// Configuration - will be loaded from backend
let API_BASE_URL = 'http://localhost:8000';  // Default fallback

// Load frontend configuration from backend on page load
async function loadFrontendConfig() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/frontend-config`);
        if (response.ok) {
            const config = await response.json();
            API_BASE_URL = config.api_base_url || API_BASE_URL;
            console.log(`Setup page config loaded: API=${API_BASE_URL}`);
        }
    } catch (error) {
        console.warn('Could not load frontend config, using defaults:', error.message);
    }
}

// Debug logging function
function debugLog(message, data = null) {
    const timestamp = new Date().toISOString().split('T')[1].split('.')[0];
    console.log(`%c[${timestamp}] [SETUP] ${message}`, 'color: #0099ff; font-weight: bold', data || '');
}

/**
 * Load configuration from .env (via backend) and auto-populate form
 */
async function loadEnvConfiguration() {
    try {
        debugLog('Checking for .env configuration...');
        const response = await fetch(`${API_BASE_URL}/api/check-env-config`);
        
        if (!response.ok) {
            debugLog('.env configuration not available or incomplete');
            return;
        }
        
        const result = await response.json();
        
        if (!result.env_configured) {
            debugLog('.env not configured, user will enter settings manually');
            return;
        }
        
        debugLog('Found .env configuration, auto-populating form...');
        
        // Auto-populate data source
        if (result.data_source) {
            const dataSourceRadio = document.querySelector(`input[name="dataSource"][value="${result.data_source}"]`);
            if (dataSourceRadio) {
                dataSourceRadio.checked = true;
                dataSourceRadio.dispatchEvent(new Event('change'));
                debugLog(`Data source set to: ${result.data_source}`);
            }
        }
        
        // Auto-populate AWS config if available
        if (result.aws_config) {
            const aws = result.aws_config;
            if (aws.bucket) document.getElementById('s3-bucket').value = aws.bucket;
            if (aws.prefix) document.getElementById('s3-prefix').value = aws.prefix;
            if (aws.region) document.getElementById('s3-region').value = aws.region;
            debugLog('AWS configuration auto-populated');
        }
        
        // Auto-populate Local config if available
        if (result.local_config && result.local_config.path) {
            document.getElementById('local-path').value = result.local_config.path;
            debugLog('Local path auto-populated');
        }
        
        // Auto-populate NOAA config if available
        if (result.noaa_config) {
            if (result.noaa_config.temp_dir) {
                document.getElementById('noaa-cache-dir').value = result.noaa_config.temp_dir;
            }
            if (result.noaa_config.cache_days) {
                document.getElementById('noaa-cache-days').value = result.noaa_config.cache_days;
            }
            debugLog('NOAA configuration auto-populated');
        }
        
        // Auto-populate Claude API key if available
        if (result.claude_api_key) {
            document.getElementById('claude-api-key').value = result.claude_api_key;
            debugLog('Claude API key auto-populated');
        }
        
        // Auto-populate output folder if available
        if (result.output_folder) {
            document.getElementById('output-folder').value = result.output_folder;
            debugLog('Output folder auto-populated');
        }
        
        debugLog('✅ Form auto-populated from .env configuration');
        showStatus('✅ Configuration loaded from .env file', 'success');
        
    } catch (error) {
        debugLog('Error loading .env configuration: ' + error.message);
        // Non-critical error, user can still enter settings manually
    }
}

// Initialize page
document.addEventListener('DOMContentLoaded', async () => {
    debugLog('═══ SETUP PAGE LOADED ═══');
    debugLog('Initializing setup page components...');
    
    // Load frontend configuration from backend first
    await loadFrontendConfig();
    debugLog(`Using API_BASE_URL: ${API_BASE_URL}`);
    
    // Load any saved configuration from .env (auto-populate form)
    await loadEnvConfiguration();
    
    initializeDataSourceSelector();
    initializeAuthMethodSelector();
    checkGPUStatus();
    
    // Check if reconfiguring
    const urlParams = new URLSearchParams(window.location.search);
    const isReconfigure = urlParams.get('reconfigure') === 'true';
    
    if (isReconfigure) {
        debugLog('⚙️ RECONFIGURE MODE - Loading existing configuration');
        // Load saved config to pre-fill form
        loadSavedConfig();
        showStatus('ℹ️ Reconfiguring - your session will be preserved', 'info');
    } else {
        // New setup - try to load any saved config
        loadSavedConfig();
    }
    
    debugLog('Setup page initialization complete');
});

/**
 * Initialize data source radio buttons
 */
function initializeDataSourceSelector() {
    const awsOption = document.getElementById('aws-option');
    const localOption = document.getElementById('local-option');
    const noaaOption = document.getElementById('noaa-option');
    const radios = document.querySelectorAll('input[name="dataSource"]');
    
    // Set initial selection
    awsOption.classList.add('selected');
    
    radios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            // Update visual selection
            document.querySelectorAll('.radio-option').forEach(opt => 
                opt.classList.remove('selected')
            );
            e.target.closest('.radio-option').classList.add('selected');
            
            // Show/hide config sections
            const value = e.target.value;
            document.getElementById('aws-config').classList.toggle('active', value === 'aws');
            document.getElementById('local-config').classList.toggle('active', value === 'local');
            document.getElementById('noaa-config').classList.toggle('active', value === 'noaa');
        });
    });
}

/**
 * Initialize auth method selector
 */
function initializeAuthMethodSelector() {
    const authMethodSelect = document.getElementById('auth-method');
    
    authMethodSelect.addEventListener('change', (e) => {
        const method = e.target.value;
        document.getElementById('credentials-section').style.display = 
            method === 'credentials' ? 'block' : 'none';
        document.getElementById('profile-section').style.display = 
            method === 'profile' ? 'block' : 'none';
    });
}

/**
 * Check GPU status
 */
async function checkGPUStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/gpu-status`);
        const data = await response.json();
        
        const gpuStatusDiv = document.getElementById('gpu-status');
        const gpuInfoDiv = document.getElementById('gpu-info');
        
        gpuStatusDiv.style.display = 'block';
        
        if (data.gpu_info.available) {
            gpuInfoDiv.innerHTML = `
                <p><strong>✅ GPU Acceleration Available</strong></p>
                <p>Type: ${data.gpu_info.type} ${data.gpu_info.backend}</p>
                <p>Devices: ${data.gpu_info.device_count} (${data.gpu_info.device_names.join(', ')})</p>
                <p style="color: #0c5460; margin-top: 5px;">${data.recommendations.message}</p>
            `;
        } else {
            gpuInfoDiv.innerHTML = `
                <p><strong>ℹ️ Using CPU Processing</strong></p>
                <p style="color: #856404;">No GPU acceleration detected. Analysis will use CPU.</p>
                <details style="margin-top: 10px;">
                    <summary style="cursor: pointer;">How to enable GPU acceleration</summary>
                    <div style="margin-top: 10px; font-size: 12px;">
                        <p><strong>For NVIDIA GPUs:</strong></p>
                        <pre>pip install --extra-index-url=https://pypi.nvidia.com cudf-cu11 cuml-cu11 cupy-cuda11x</pre>
                        
                        <p style="margin-top: 10px;"><strong>For AMD GPUs:</strong></p>
                        <pre>pip install cupy-rocm-5-0
# or
pip install pyhip</pre>
                    </div>
                </details>
            `;
        }
    } catch (error) {
        console.error('Failed to check GPU status:', error);
    }
}

/**
 * Gather configuration from form
 */
function gatherConfig() {
    const dataSource = document.querySelector('input[name="dataSource"]:checked').value;
    
    const config = {
        data_source: dataSource,
        claude_api_key: document.getElementById('claude-api-key').value,
        output_folder: document.getElementById('output-folder').value,
        user_name: document.getElementById('user-name').value,
        date_range: {
            min: '2024-10-15',
            max: '2025-03-30'
        }
    };
    
    if (dataSource === 'aws') {
        const authMethod = document.getElementById('auth-method').value;
        config.aws = {
            bucket: document.getElementById('s3-bucket').value,
            prefix: document.getElementById('s3-prefix').value,
            region: document.getElementById('s3-region').value,
            auth_method: authMethod
        };
        
        if (authMethod === 'credentials') {
            config.aws.access_key = document.getElementById('access-key').value;
            config.aws.secret_key = document.getElementById('secret-key').value;
            config.aws.session_token = document.getElementById('session-token').value;
        } else if (authMethod === 'profile') {
            config.aws.profile_name = document.getElementById('profile-name').value || 'default';
        }
    } else if (dataSource === 'local') {
        const localPath = document.getElementById('local-path');
        const fileFormat = document.getElementById('file-format');
        
        config.local = {
            path: localPath ? localPath.value : '',
            file_format: fileFormat ? fileFormat.value : 'auto'
        };
    } else if (dataSource === 'noaa') {
        const cacheDirEl = document.getElementById('noaa-cache-dir');
        const cacheDaysEl = document.getElementById('noaa-cache-days');
        
        config.noaa = {};
        
        if (cacheDirEl && cacheDirEl.value.trim()) {
            config.noaa.temp_dir = cacheDirEl.value.trim();
        }
        if (cacheDaysEl && cacheDaysEl.value) {
            config.noaa.cache_days = parseInt(cacheDaysEl.value);
        }
    }
    
    return config;
}

/**
 * Validate that a path is a full absolute path
 */
function isValidFullPath(path) {
    if (!path || !path.trim()) {
        return false;
    }
    
    const trimmedPath = path.trim();
    
    // Check for Windows absolute path (starts with drive letter)
    if (trimmedPath.match(/^[A-Za-z]:\\/)) {
        return true;
    }
    
    // Check for Unix/Mac absolute path (starts with /)
    if (trimmedPath.startsWith('/')) {
        return true;
    }
    
    // Check for UNC path (Windows network path)
    if (trimmedPath.startsWith('\\\\')) {
        return true;
    }
    
    return false;
}

/**
 * Validate configuration
 */
function validateConfig(config) {
    if (!config.claude_api_key) {
        return { valid: false, message: 'Claude API key is required' };
    }
    
    // Output folder is optional - backend will use default if not specified
    // (AISDS_Output in user's Downloads folder)
    
    if (config.data_source === 'aws') {
        if (!config.aws.bucket) {
            return { valid: false, message: 'S3 bucket name is required' };
        }
        if (config.aws.auth_method === 'credentials') {
            if (!config.aws.access_key || !config.aws.secret_key) {
                return { valid: false, message: 'AWS credentials are required' };
            }
        }
    } else if (config.data_source === 'local') {
        if (!config.local.path) {
            return { valid: false, message: 'Local data folder path is required' };
        }
        
        // Validate that the path is a full absolute path
        if (!isValidFullPath(config.local.path)) {
            return { 
                valid: false, 
                message: 'Local data folder path must be a full absolute path.\n\n' +
                         'Windows: Start with drive letter (e.g., C:\\AIS_Data)\n' +
                         'Mac/Linux: Start with / (e.g., /home/user/ais_data)'
            };
        }
    } else if (config.data_source === 'noaa') {
        // NOAA requires no credentials, only optional cache directory
        // If cache directory is provided, validate it's a full path
        if (config.noaa && config.noaa.temp_dir && config.noaa.temp_dir.trim()) {
            if (!isValidFullPath(config.noaa.temp_dir)) {
                return {
                    valid: false,
                    message: 'Cache directory must be a full absolute path.\n\n' +
                             'Windows: Start with drive letter (e.g., C:\\AIS_Cache)\n' +
                             'Mac/Linux: Start with / (e.g., /tmp/ais_cache)\n\n' +
                             'Or leave blank to use default temp directory.'
                };
            }
        }
    }
    
    // Validate output folder if provided
    if (config.output_folder && config.output_folder.trim()) {
        if (!isValidFullPath(config.output_folder)) {
            return {
                valid: false,
                message: 'Output folder path must be a full absolute path.\n\n' +
                         'Windows: Start with drive letter (e.g., C:\\Users\\YourName\\Output)\n' +
                         'Mac/Linux: Start with / (e.g., /home/user/output)\n\n' +
                         'Or leave blank to use default location.'
            };
        }
    }
    
    return { valid: true };
}

/**
 * Show status message
 */
function showStatus(message, type) {
    const statusEl = document.getElementById('status-message');
    statusEl.textContent = message;
    statusEl.className = `status-message ${type}`;
    statusEl.style.display = 'block';
    
    if (type === 'success') {
        setTimeout(() => {
            statusEl.style.display = 'none';
        }, 5000);
    }
}

/**
 * Browse for output folder
 * Uses File System Access API to select a folder and attempts to resolve full path
 * Since browsers don't expose full paths directly, we try to build it from the directory handle
 */
async function browseOutputFolder() {
    const input = document.getElementById('output-folder');
    const currentPath = input.value;
    
    try {
        if ('showDirectoryPicker' in window) {
            const dirHandle = await window.showDirectoryPicker({
                mode: 'read'  // Read mode - we'll use this as a starting point
            });
            
            const folderName = dirHandle.name;
            
            // Try to resolve the full path by walking up the directory tree
            let fullPath = '';
            try {
                // Attempt to build path by querying parent directories
                const pathParts = [folderName];
                let currentHandle = dirHandle;
                
                // Try to walk up to root (limited to reasonable depth)
                for (let i = 0; i < 20; i++) {
                    try {
                        // Try to get parent - this may not work in all browsers
                        if (currentHandle.getParent) {
                            const parentHandle = await currentHandle.getParent();
                            if (parentHandle) {
                                const parentName = parentHandle.name || '';
                                if (parentName && parentName !== currentHandle.name) {
                                    pathParts.unshift(parentName);
                                    currentHandle = parentHandle;
                                } else {
                                    break;
                                }
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    } catch (e) {
                        // Can't get parent, stop here
                        break;
                    }
                }
                
                // Construct path based on platform
                if (navigator.platform.toLowerCase().includes('win')) {
                    // Windows - check if we have a drive letter
                    if (pathParts.length > 0 && pathParts[0].match(/^[A-Z]:$/)) {
                        fullPath = pathParts.join('\\');
                    } else {
                        // No drive letter - construct from common Windows structure
                        fullPath = `C:\\${pathParts.join('\\')}`;
                    }
                } else {
                    // Mac/Linux - start with root
                    fullPath = '/' + pathParts.join('/');
                }
            } catch (pathError) {
                // If path resolution fails, use folder name and prompt user
                debugLog('Could not resolve full path, using folder name', pathError);
                fullPath = folderName;
            }
            
            // If we couldn't get a proper full path, show a warning
            const isFullPath = (navigator.platform.toLowerCase().includes('win') && fullPath.includes(':\\')) ||
                               (!navigator.platform.toLowerCase().includes('win') && fullPath.startsWith('/'));
            
            if (!isFullPath) {
                // Show alert asking user to enter full path manually
                const message = 
                    `📁 FOLDER SELECTED: ${folderName}\n\n` +
                    `⚠️ IMPORTANT: Please enter the FULL ABSOLUTE PATH manually.\n\n` +
                    `The browser cannot access the full file system path for security reasons.\n\n` +
                    `Please type or paste the complete path:\n\n` +
                    `Windows Examples:\n` +
                    `• C:\\Users\\YourName\\Documents\\AIS_Output\n` +
                    `• C:\\Users\\YourName\\Downloads\\AISDS_Output\n` +
                    `• D:\\Output\\AIS\n\n` +
                    `Mac/Linux Examples:\n` +
                    `• /Users/yourname/Documents/AIS_Output\n` +
                    `• /home/yourname/output\n\n` +
                    `💡 Tip: Leave blank to use default: Downloads/AISDS_Output\n\n` +
                    `The path must be absolute (start with drive letter on Windows or / on Mac/Linux).`;
                
                alert(message);
                
                // Clear and focus input for manual entry
                input.value = '';
                input.focus();
                input.placeholder = 'Enter full absolute path (or leave blank for default)';
            } else {
                // Set the resolved path
                input.value = fullPath;
                
                // Show success message with verification prompt
                showStatus(
                    `✅ Folder Selected: ${folderName}\n\n` +
                    `Resolved Path: ${fullPath}\n\n` +
                    `⚠️ Please VERIFY this is the correct full path.\n\n` +
                    `If the path looks incorrect, edit it manually.\n\n` +
                    `💡 Tip: Leave blank to use default: Downloads/AISDS_Output`,
                    'success'
                );
                
                debugLog('Output folder selected', { name: folderName, resolvedPath: fullPath });
                
                // Focus and select so user can verify/edit
                input.focus();
                input.select();
            }
            
        } else {
            // Fallback: Browser doesn't support folder picker
            const message = 
                '📁 OUTPUT FOLDER SELECTION\n\n' +
                'Your browser doesn\'t support folder selection.\n\n' +
                'Please manually enter the FULL ABSOLUTE PATH where you want to save results:\n\n' +
                'Windows Examples:\n' +
                '• C:\\Users\\YourName\\Documents\\AIS_Output\n' +
                '• C:\\Users\\YourName\\Downloads\\AISDS_Output\n\n' +
                'Mac/Linux Examples:\n' +
                '• /Users/yourname/Documents/AIS_Output\n' +
                '• /home/yourname/output\n\n' +
                '💡 Tip: Leave blank to use default location:\n' +
                '   Downloads/AISDS_Output\n\n' +
                '⚠️ The path must be absolute (start with drive letter or /)\n\n' +
                'The folder will be created automatically if it doesn\'t exist.';
            
            alert(message);
            
            // Focus the input so user can type
            input.focus();
            input.placeholder = 'Enter full absolute path (or leave blank for default)';
            
            // If there's no current value, select all; otherwise place cursor at end
            if (!currentPath) {
                input.select();
            } else {
                input.setSelectionRange(currentPath.length, currentPath.length);
            }
            
            debugLog('Output folder browse - showing manual entry instructions');
        }
    } catch (err) {
        if (err.name === 'AbortError') {
            // User cancelled - this is normal
            debugLog('Output folder selection cancelled by user');
        } else {
            console.error('Folder selection error:', err);
            showStatus('❌ Folder selection failed. Please enter the full path manually or leave blank for default.', 'error');
            input.focus();
            input.placeholder = 'Enter full absolute path (or leave blank for default)';
        }
    }
}

/**
 * Browse for local data folder
 * Uses File System Access API to select a folder and attempts to resolve full path
 * Since browsers don't expose full paths directly, we try to build it from the directory handle
 */
async function browseLocalFolder() {
    const input = document.getElementById('local-path');
    const currentValue = input.value;
    
    try {
        if ('showDirectoryPicker' in window) {
            // Modern browsers with File System Access API
            const dirHandle = await window.showDirectoryPicker({
                mode: 'read'  // Read-only access for data folder
            });
            
            // Get folder name
            const folderName = dirHandle.name;
            
            // Try to resolve the full path by walking up the directory tree
            let fullPath = '';
            try {
                // Attempt to build path by querying parent directories
                const pathParts = [folderName];
                let currentHandle = dirHandle;
                
                // Try to walk up to root (limited to reasonable depth)
                for (let i = 0; i < 20; i++) {
                    try {
                        // Try to get parent - this may not work in all browsers
                        if (currentHandle.getParent) {
                            const parentHandle = await currentHandle.getParent();
                            if (parentHandle) {
                                const parentName = parentHandle.name || '';
                                if (parentName && parentName !== currentHandle.name) {
                                    pathParts.unshift(parentName);
                                    currentHandle = parentHandle;
                                } else {
                                    break;
                                }
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    } catch (e) {
                        // Can't get parent, stop here
                        break;
                    }
                }
                
                // Construct path based on platform
                if (navigator.platform.toLowerCase().includes('win')) {
                    // Windows - check if we have a drive letter
                    if (pathParts.length > 0 && pathParts[0].match(/^[A-Z]:$/)) {
                        fullPath = pathParts.join('\\');
                    } else {
                        // No drive letter - construct from common Windows structure
                        fullPath = `C:\\${pathParts.join('\\')}`;
                    }
                } else {
                    // Mac/Linux - start with root
                    fullPath = '/' + pathParts.join('/');
                }
            } catch (pathError) {
                // If path resolution fails, use folder name and prompt user
                debugLog('Could not resolve full path, using folder name', pathError);
                fullPath = folderName;
            }
            
            // If we couldn't get a proper full path, show a warning
            const isFullPath = (navigator.platform.toLowerCase().includes('win') && fullPath.includes(':\\')) ||
                               (!navigator.platform.toLowerCase().includes('win') && fullPath.startsWith('/'));
            
            if (!isFullPath) {
                // Show alert asking user to enter full path manually
                const message = 
                    `📁 FOLDER SELECTED: ${folderName}\n\n` +
                    `⚠️ IMPORTANT: Please enter the FULL ABSOLUTE PATH manually.\n\n` +
                    `The browser cannot access the full file system path for security reasons.\n\n` +
                    `Please type or paste the complete path:\n\n` +
                    `Windows Examples:\n` +
                    `• C:\\AIS_Data\n` +
                    `• C:\\Users\\YourName\\Documents\\ais_data\n` +
                    `• D:\\Data\\AIS\n\n` +
                    `Mac/Linux Examples:\n` +
                    `• /Users/yourname/ais_data\n` +
                    `• /home/yourname/Documents/ais_data\n` +
                    `• /mnt/data/ais\n\n` +
                    `The path must be absolute (start with drive letter on Windows or / on Mac/Linux).`;
                
                alert(message);
                
                // Clear and focus input for manual entry
                input.value = '';
                input.focus();
                input.placeholder = 'Enter full absolute path (e.g., C:\\AIS_Data or /home/user/ais_data)';
            } else {
                // Set the resolved path
                input.value = fullPath;
                
                // Show success message with verification prompt
                showStatus(
                    `✅ Folder Selected: ${folderName}\n\n` +
                    `Resolved Path: ${fullPath}\n\n` +
                    `⚠️ Please VERIFY this is the correct full path.\n\n` +
                    `If the path looks incorrect, edit it manually.\n\n` +
                    `The folder should contain data files named by date (e.g., 2024-10-15.parquet)`,
                    'success'
                );
                
                debugLog('Local folder selected', { name: folderName, resolvedPath: fullPath });
                
                // Focus and select so user can verify/edit
                input.focus();
                input.select();
            }
            
        } else {
            // Fallback: Browser doesn't support folder picker
            alert(
                '📁 DATA FOLDER SELECTION\n\n' +
                'Your browser doesn\'t support folder selection.\n\n' +
                'Please manually enter the FULL ABSOLUTE PATH to your data folder:\n\n' +
                'Windows Examples:\n' +
                '• C:\\AIS_Data\n' +
                '• C:\\Users\\YourName\\ais_data\n' +
                '• D:\\Data\\AIS\n\n' +
                'Mac/Linux Examples:\n' +
                '• /Users/yourname/ais_data\n' +
                '• /home/yourname/Documents/ais_data\n\n' +
                'The folder should contain parquet or CSV files named by date:\n' +
                '• 2024-10-15.parquet (or .csv)\n' +
                '• ais-2024-10-15.parquet (or .csv)\n\n' +
                '⚠️ The path must be absolute (start with drive letter or /)'
            );
            input.focus();
            input.placeholder = 'Enter full absolute path (e.g., C:\\AIS_Data or /home/user/ais_data)';
        }
    } catch (err) {
        if (err.name === 'AbortError') {
            // User cancelled - this is normal, no error needed
            debugLog('Local folder selection cancelled by user');
        } else {
            console.error('Folder selection error:', err);
            showStatus('❌ Folder selection failed. Please enter the full path manually.', 'error');
            input.focus();
            input.placeholder = 'Enter full absolute path manually';
        }
    }
}

/**
 * Test settings (AWS or Local)
 */
async function testSettings() {
    const config = gatherConfig();
    
    if (config.data_source === 'aws') {
        return await testAWSConnection();
    } else if (config.data_source === 'local') {
        return await testLocalPath();
    }
}

/**
 * Test AWS connection
 */
async function testAWSConnection() {
    const config = gatherConfig();
    
    // Validate AWS config
    if (!config.aws || !config.aws.bucket) {
        showStatus('❌ Please fill in AWS S3 configuration', 'error');
        return;
    }
    
    if (config.aws.auth_method === 'credentials' && (!config.aws.access_key || !config.aws.secret_key)) {
        showStatus('❌ AWS credentials are required', 'error');
        return;
    }
    
    showStatus('🔍 Testing AWS S3 connection...', 'info');
    debugLog('Testing AWS connection', config.aws);
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/test-aws`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config.aws)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            // Build detailed success message
            let message = `✅ AWS Connection Successful!\n\n`;
            message += `📁 Bucket: ${result.bucket}\n`;
            message += `📂 Prefix: ${result.prefix || '(none)'}\n`;
            message += `📊 Files Found: ${result.object_count || 0} objects\n`;
            message += `📄 Test File: ${result.file_found}\n`;
            message += `📋 Format: ${result.file_type?.toUpperCase() || 'UNKNOWN'}\n`;
            message += `📊 Columns: ${result.column_count || 0} columns\n`;
            
            if (result.columns && result.columns.length > 0) {
                message += `\n📋 Column Names:\n${result.columns.slice(0, 10).join(', ')}`;
                if (result.columns.length > 10) {
                    message += ` ... (+${result.columns.length - 10} more)`;
                }
            }
            
            if (result.sample_data && result.sample_data.length > 0) {
                message += `\n\n📊 Sample Data (First ${result.row_count_sample || 5} rows):\n`;
                message += `\n${JSON.stringify(result.sample_data, null, 2)}`;
            }
            
            showStatus(message, 'success');
            debugLog('AWS test passed', result);
        } else {
            showStatus(`❌ AWS Test Failed: ${result.error}`, 'error');
            debugLog('AWS test failed', result);
        }
    } catch (error) {
        showStatus(`❌ Connection Error: ${error.message}`, 'error');
        debugLog('AWS test error', error);
    }
}

/**
 * Test local path
 */
async function testLocalPath() {
    const config = gatherConfig();
    
    // Validate local config
    if (!config.local || !config.local.path) {
        showStatus('❌ Please enter a local data folder path', 'error');
        return;
    }
    
    showStatus('🔍 Testing local path...', 'info');
    debugLog('Testing local path', config.local);
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/test-local-path`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ local_directory: config.local.path })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            // Build detailed success message
            let message = `✅ Local Path Valid!\n\n`;
            message += `📁 Directory: ${config.local.path}\n`;
            message += `📊 Total Files: ${result.total_files || 0} files\n`;
            message += `📄 Test File: ${result.file_found}\n`;
            message += `📋 Format: ${result.file_type?.toUpperCase() || 'UNKNOWN'}\n`;
            message += `📊 Columns: ${result.column_count || 0} columns\n`;
            
            if (result.columns && result.columns.length > 0) {
                message += `\n📋 Column Names:\n${result.columns.slice(0, 10).join(', ')}`;
                if (result.columns.length > 10) {
                    message += ` ... (+${result.columns.length - 10} more)`;
                }
            }
            
            if (result.sample_data && result.sample_data.length > 0) {
                message += `\n\n📊 Sample Data (First ${result.row_count_sample || 5} rows):\n`;
                message += `\n${JSON.stringify(result.sample_data, null, 2)}`;
            }
            
            showStatus(message, 'success');
            debugLog('Local path test passed', result);
        } else {
            showStatus(`❌ Local Path Test Failed: ${result.error}`, 'error');
            debugLog('Local path test failed', result);
        }
    } catch (error) {
        showStatus(`❌ Test Error: ${error.message}`, 'error');
        debugLog('Local path test error', error);
    }
}

/**
 * Test connection (legacy - redirects to testSettings)
 */
async function testConnection() {
    return await testSettings();
}

/**
 * Save configuration and continue to main app
 */
async function saveAndContinue() {
    debugLog('═══ SAVE AND CONTINUE CLICKED ═══');
    
    const config = gatherConfig();
    debugLog('Gathered configuration:', {
        data_source: config.data_source,
        has_claude_key: !!config.claude_api_key,
        claude_key_length: config.claude_api_key ? config.claude_api_key.length : 0,
        has_aws: !!config.aws,
        has_local: !!config.local,
        output_folder: config.output_folder,
        user_name: config.user_name
    });
    
    // Validate
    const validation = validateConfig(config);
    debugLog('Validation result:', validation);
    
    if (!validation.valid) {
        debugLog('❌ Validation failed:', validation.message);
        showStatus(`❌ ${validation.message}`, 'error');
        return;
    }
    
    showStatus('⏳ Initializing system...', 'info');
    debugLog('Sending configuration to backend...');
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/setup`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        
        debugLog('Backend response status:', response.status);
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP ${response.status}: ${errorText || response.statusText}`);
        }
        
        const result = await response.json();
        debugLog('Backend response data:', result);
        
        if (result.success) {
            debugLog('✅ Backend setup successful!');
            debugLog('Session ID:', result.session_id);
            
            // Store session info
            localStorage.setItem('ais_session_id', result.session_id);
            debugLog('Stored session ID in localStorage');
            
            localStorage.setItem('ais_config', JSON.stringify(config));
            debugLog('Stored config in localStorage');
            
            localStorage.setItem('ais_date_range', JSON.stringify(result.date_range));
            debugLog('Stored date range in localStorage');
            
            // Verify what was stored
            debugLog('Verification - localStorage contents:', {
                session_id: localStorage.getItem('ais_session_id'),
                config_exists: !!localStorage.getItem('ais_config'),
                config_length: localStorage.getItem('ais_config')?.length
            });
            
            showStatus('✅ Configuration saved! Redirecting...', 'success');
            
            // Redirect to main app with fast-load parameter
            debugLog('Redirecting to index.html in 1500ms...');
            setTimeout(() => {
                debugLog('REDIRECT: Navigating to index.html NOW (with fast-load param)');
                window.location.href = 'index.html?from=setup';
            }, 1500);
        } else {
            debugLog('❌ Backend setup failed:', result.error);
            showStatus(`❌ Setup failed: ${result.error}`, 'error');
        }
    } catch (error) {
        debugLog('❌ Exception during save:', error);
        showStatus(`❌ Error: ${error.message}`, 'error');
    }
}

/**
 * Load saved configuration
 */
function loadSavedConfig() {
    const savedConfig = localStorage.getItem('ais_config');
    if (!savedConfig) return;
    
    try {
        const config = JSON.parse(savedConfig);
        
        // Restore data source selection
        const dataSourceRadio = document.querySelector(`input[name="dataSource"][value="${config.data_source}"]`);
        if (dataSourceRadio) {
            dataSourceRadio.checked = true;
            dataSourceRadio.dispatchEvent(new Event('change'));
        }
        
        // Restore AWS config
        if (config.aws) {
            document.getElementById('s3-bucket').value = config.aws.bucket || '';
            document.getElementById('s3-prefix').value = config.aws.prefix || '';
            document.getElementById('s3-region').value = config.aws.region || 'us-east-1';
            document.getElementById('auth-method').value = config.aws.auth_method || 'credentials';
            
            if (config.aws.auth_method === 'credentials') {
                document.getElementById('access-key').value = config.aws.access_key || '';
                // Don't restore secret key for security
            } else if (config.aws.auth_method === 'profile') {
                document.getElementById('profile-name').value = config.aws.profile_name || 'default';
            }
        }
        
        // Restore local config
        if (config.local) {
            document.getElementById('local-path').value = config.local.path || '';
            document.getElementById('file-format').value = config.local.file_format || 'auto';
        }
        
        // Restore output folder
        if (config.output_folder) {
            document.getElementById('output-folder').value = config.output_folder;
        }
        
        // Restore user name
        if (config.user_name) {
            document.getElementById('user-name').value = config.user_name;
        }
        
        // Don't restore Claude API key for security
        
    } catch (error) {
        console.error('Error loading saved config:', error);
    }
}


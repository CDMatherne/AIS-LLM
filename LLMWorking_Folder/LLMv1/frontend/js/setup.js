/**
 * AIS Law Enforcement Assistant - Setup Page JavaScript
 */

// Configuration
const API_BASE_URL = 'http://localhost:8000';

// Debug logging function
function debugLog(message, data = null) {
    const timestamp = new Date().toISOString().split('T')[1].split('.')[0];
    console.log(`%c[${timestamp}] [SETUP] ${message}`, 'color: #0099ff; font-weight: bold', data || '');
}

// Initialize page
document.addEventListener('DOMContentLoaded', () => {
    debugLog('═══ SETUP PAGE LOADED ═══');
    debugLog('Initializing setup page components...');
    
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
    } else {
        config.local = {
            path: document.getElementById('local-path').value,
            file_format: document.getElementById('file-format').value
        };
    }
    
    return config;
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
 * 
 * Similar to SFD_GUI.py's browse_output_directory() - provides a simple folder selection experience.
 * 
 * NOTE: Due to browser security restrictions, we cannot use showDirectoryPicker() for write access.
 * This function attempts to use the File System Access API in read mode to help users select a folder,
 * then they can verify/complete the full path for write access.
 */
async function browseOutputFolder() {
    const input = document.getElementById('output-folder');
    const currentPath = input.value;
    
    try {
        // Try to use File System Access API (read mode) to help user select folder
        // Even though we need write access, this helps them pick the right location
        if ('showDirectoryPicker' in window) {
            try {
                const dirHandle = await window.showDirectoryPicker({
                    mode: 'read'  // Read mode - we'll use this as a starting point
                });
                
                const folderName = dirHandle.name;
                
                // Try to construct a reasonable path
                // On Windows, try common locations
                let suggestedPath = '';
                
                if (navigator.platform.toLowerCase().includes('win')) {
                    // Windows - try to construct path
                    if (currentPath && currentPath.includes('\\')) {
                        // Use existing path structure
                        const parts = currentPath.split('\\');
                        parts[parts.length - 1] = folderName;
                        suggestedPath = parts.join('\\');
                    } else {
                        // Default to Downloads folder (common Windows location)
                        suggestedPath = `C:\\Users\\YourName\\Downloads\\${folderName}`;
                    }
                } else {
                    // Mac/Linux
                    if (currentPath && currentPath.includes('/')) {
                        const parts = currentPath.split('/');
                        parts[parts.length - 1] = folderName;
                        suggestedPath = parts.join('/');
                    } else {
                        // Default to home directory
                        suggestedPath = `/home/yourname/${folderName}`;
                    }
                }
                
                // Set the suggested path
                input.value = suggestedPath;
                
                // Show helpful message
                showStatus(
                    `📁 Folder Selected: ${folderName}\n\n` +
                    `Suggested path: ${suggestedPath}\n\n` +
                    `⚠️ Please verify or edit the FULL PATH above.\n\n` +
                    `The browser shows the folder name, but you may need to adjust the full path.\n\n` +
                    `💡 Tip: Leave blank to use default: Downloads/AISDS_Output`,
                    'info'
                );
                
                // Focus and select so user can edit
                input.focus();
                input.select();
                
                debugLog('Output folder selected via directory picker', { name: folderName, suggestedPath });
                
            } catch (err) {
                if (err.name === 'AbortError') {
                    // User cancelled - this is normal
                    debugLog('Output folder selection cancelled by user');
                    return;
                }
                throw err; // Re-throw to fallback handler
            }
        } else {
            // Fallback: Browser doesn't support folder picker
            throw new Error('Folder picker not supported');
        }
    } catch (error) {
        // Fallback: Show instructions for manual entry (like SFD_GUI.py's simple approach)
        const message = 
            '📁 OUTPUT FOLDER SELECTION\n\n' +
            'Please enter the FULL PATH where you want to save results.\n\n' +
            'Examples:\n' +
            '• Windows: C:\\Users\\YourName\\Documents\\AIS_Output\n' +
            '• Windows: C:\\Users\\YourName\\Downloads\\AISDS_Output\n' +
            '• Mac: /Users/yourname/Documents/AIS_Output\n' +
            '• Linux: /home/yourname/ais_output\n\n' +
            '💡 Tip: Leave blank to use default location:\n' +
            '   Downloads/AISDS_Output\n\n' +
            'The folder will be created automatically if it doesn\'t exist.';
        
        alert(message);
        
        // Focus the input so user can type
        input.focus();
        
        // If there's no current value, select all; otherwise place cursor at end
        if (!currentPath) {
            input.select();
        } else {
            input.setSelectionRange(currentPath.length, currentPath.length);
        }
        
        debugLog('Output folder browse - showing manual entry instructions');
    }
}

/**
 * Browse for local data folder
 * Similar to SFD_GUI.py's browse_data_directory() - simple folder selection
 * Uses File System Access API to select a folder (read-only access)
 */
async function browseLocalFolder() {
    const input = document.getElementById('local-path');
    const currentValue = input.value;
    
    try {
        if ('showDirectoryPicker' in window) {
            // Modern browsers with File System Access API
            // Similar to tkinter's filedialog.askdirectory() - opens native folder picker
            const dirHandle = await window.showDirectoryPicker({
                mode: 'read'  // Read-only access for data folder
            });
            
            // Get folder name
            const folderName = dirHandle.name;
            
            // Try to construct full path similar to SFD_GUI.py's approach
            // If there's already a path, try to preserve parent directory structure
            let fullPath = '';
            
            if (currentValue && currentValue.includes('\\')) {
                // Windows path - preserve parent structure
                const parts = currentValue.split('\\');
                parts[parts.length - 1] = folderName;
                fullPath = parts.join('\\');
            } else if (currentValue && currentValue.includes('/')) {
                // Unix/Mac path - preserve parent structure
                const parts = currentValue.split('/');
                parts[parts.length - 1] = folderName;
                fullPath = parts.join('/');
            } else {
                // No existing path - try to construct reasonable default
                if (navigator.platform.toLowerCase().includes('win')) {
                    // Windows - try common data locations
                    const userProfile = 'C:\\Users\\YourName'; // Placeholder
                    fullPath = `${userProfile}\\${folderName}`;
                } else {
                    // Mac/Linux
                    const homeDir = '/home/yourname'; // Placeholder
                    fullPath = `${homeDir}/${folderName}`;
                }
            }
            
            // Set the path (user can edit if needed)
            input.value = fullPath;
            
            // Show success message
            showStatus(
                `✅ Folder Selected: ${folderName}\n\n` +
                `Path: ${fullPath}\n\n` +
                `💡 Please verify the full path above and adjust if needed.\n\n` +
                `The folder should contain data files named by date (e.g., 2024-10-15.parquet)`,
                'success'
            );
            
            debugLog('Local folder selected', { name: folderName, path: fullPath });
            
            // Focus and select so user can verify/edit
            input.focus();
            input.select();
            
        } else {
            // Fallback: Browser doesn't support folder picker
            // Show instructions similar to SFD_GUI.py's simple approach
            alert(
                '📁 DATA FOLDER SELECTION\n\n' +
                'Your browser doesn\'t support folder selection.\n\n' +
                'Please manually enter the FULL PATH to your data folder:\n\n' +
                'Examples:\n' +
                '• Windows: C:\\AIS_Data\n' +
                '• Windows: C:\\Users\\YourName\\ais_data\n' +
                '• Mac: /Users/yourname/ais_data\n' +
                '• Linux: /home/yourname/ais_data\n\n' +
                'The folder should contain parquet or CSV files named by date:\n' +
                '• 2024-10-15.parquet (or .csv)\n' +
                '• ais-2024-10-15.parquet (or .csv)'
            );
            input.focus();
        }
    } catch (err) {
        if (err.name === 'AbortError') {
            // User cancelled - this is normal, no error needed
            debugLog('Local folder selection cancelled by user');
        } else {
            console.error('Folder selection error:', err);
            showStatus('❌ Folder selection failed. Please enter the path manually.', 'error');
            input.focus();
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
            
            // Redirect to main app
            debugLog('Redirecting to index.html in 1500ms...');
            setTimeout(() => {
                debugLog('REDIRECT: Navigating to index.html NOW');
                window.location.href = 'index.html';
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


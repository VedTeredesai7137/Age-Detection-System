# Deployment Troubleshooting Guide

## Build Issues Fixed

### Problem 1: setuptools.build_meta Import Error
**Solution**: Updated requirements.txt with compatible versions and added setuptools explicitly.

### Problem 2: Python Version Compatibility  
**Solution**: Changed runtime.txt from 3.11.9 to 3.11.7 (more stable on Render).

### Problem 3: Dependency Conflicts
**Solution**: Used tested, compatible package versions instead of latest.

## Current Configuration

### Runtime: Python 3.11.7
- More stable than 3.11.9 on Render
- Better compatibility with TensorFlow 2.12.0

### Dependencies (requirements.txt):
```
Flask==2.3.3
Werkzeug==2.3.7
opencv-python-headless==4.7.1.72  # Headless for servers
numpy==1.23.5                      # Compatible with TF 2.12
tensorflow==2.12.0                 # Stable version
keras==2.12.0                      # Matches TensorFlow
mtcnn==0.1.1
Pillow==9.5.0
moviepy==1.0.3
gunicorn==20.1.0                   # Proven stable
setuptools==67.8.0                 # Fixes build_meta issue
wheel==0.40.0
```

### Build Process:
1. Upgrade pip, setuptools, wheel first
2. Install requirements with exact versions
3. Create necessary directories

## Camera Functionality Explanation

### JavaScript Hostname Check (camera.html lines 293-297):

```javascript
if (window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
    updateStatus('Camera not available in deployment environment', 'disconnected');
    updateButtons(true, false);
    return;
}
```

### Why This is Necessary:
1. **Server Limitation**: Web servers don't have camera hardware
2. **User Experience**: Prevents confusing error messages
3. **Graceful Degradation**: App still works for image upload
4. **Clear Communication**: Users understand the limitation

### Will This Cause Errors? NO!
- ✅ **Safe Detection**: Uses standard web API
- ✅ **Graceful Handling**: Shows informative message
- ✅ **No Exceptions**: Prevents actual camera access attempts
- ✅ **Main Feature Works**: Image upload still functional

### Alternative Detection Methods:
```javascript
// Method 1: Check for production environment
if (window.location.protocol === 'https:' && 
    !window.location.hostname.includes('localhost')) {
    // Disable camera
}

// Method 2: Check for specific domains
if (window.location.hostname.includes('render.com') || 
    window.location.hostname.includes('herokuapp.com')) {
    // Disable camera
}

// Method 3: Environment variable (requires server-side rendering)
// Not recommended for client-side detection
```

## Deployment Steps (Updated):

1. **Update Code**: Ensure all changes are committed
2. **Render Configuration**:
   - Build Command: `bash build.sh`
   - Start Command: (leave empty, uses Procfile)
   - Runtime: Python 3.11.7
3. **Deploy**: Should build successfully now

## Common Issues & Solutions:

### Issue: "Cannot import setuptools.build_meta"
**Solution**: ✅ Fixed with setuptools==67.8.0 in requirements.txt

### Issue: TensorFlow version conflicts
**Solution**: ✅ Using TensorFlow 2.12.0 with Python 3.11.7

### Issue: OpenCV not working on server
**Solution**: ✅ Using opencv-python-headless==4.7.1.72

### Issue: Camera not working in deployment
**Solution**: ✅ Intentionally disabled with clear user message

## Performance Optimizations:

### Gunicorn Settings:
```
--timeout 300           # 5 minutes for model loading
--worker-class sync     # Simple, memory-efficient
--workers 1            # Single worker for ML model
--max-requests 100     # Restart worker periodically
--preload              # Load app before forking
```

### Memory Management:
- Single worker to avoid model duplication
- Preload app to reduce startup time per request
- Regular worker restart to prevent memory leaks

## Monitoring:

### Success Indicators:
- Build completes without errors
- App starts successfully
- Image upload works
- Age detection produces results
- Camera gracefully shows "not available" message

### Log Monitoring:
```bash
# Check these in Render logs:
[INFO] Loading models on startup...
[INFO] Models loaded successfully!
[INFO] Starting Flask server...
```

## Alternative Configurations:

If the current setup fails, try these alternatives:

### Alternative 1: Minimal Requirements
Remove moviepy if video processing isn't needed:
```
Flask==2.3.3
opencv-python-headless==4.7.1.72
numpy==1.23.5
tensorflow==2.12.0
mtcnn==0.1.1
Pillow==9.5.0
gunicorn==20.1.0
```

### Alternative 2: Different Python Version
Try Python 3.10.12 in runtime.txt if 3.11.7 fails

### Alternative 3: Simplified Build
Build command: `pip install -r requirements.txt`
(Remove bash build.sh if it causes issues)
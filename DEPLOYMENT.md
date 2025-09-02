# Render Deployment Configuration

This document contains all the settings and information needed to deploy the Age Detection SSR-NET application on Render.

## Render Service Settings

### Basic Configuration
- **Service Type**: Web Service
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn --bind 0.0.0.0:$PORT app:app --timeout 120 --worker-class sync --workers 1`
- **Root Directory**: Leave empty (use repository root)

### Environment Settings
- **Branch**: main (or your default branch)
- **Runtime**: Python 3.11.9 (specified in runtime.txt)
- **Region**: Choose closest to your users

### Environment Variables
The following environment variables will be automatically set by Render:
- `PORT`: Automatically provided by Render
- `RENDER`: Set to indicate deployment environment (handled in app.py)

No additional environment variables are required for basic functionality.

### Advanced Settings
- **Health Check Path**: `/` (optional, uses default route)
- **Pre-Deploy Command**: None required
- **Docker**: Not used (using buildpack deployment)

## File Structure for Deployment

The application includes the following key files for Render deployment:

```
/
├── app.py                 # Main Flask application
├── Procfile              # Process file for Render
├── requirements.txt      # Python dependencies
├── runtime.txt          # Python version specification
├── .slugignore          # Files to exclude from deployment
├── templates/           # HTML templates
│   ├── index.html
│   └── camera.html
├── demo/               # Model architecture
│   └── SSRNET_model.py
├── pre-trained/        # Pre-trained model weights
│   ├── imdb/
│   ├── wiki/
│   └── ...
├── uploads/            # Directory for uploaded images
│   └── .gitkeep
└── results/            # Directory for processed results
    └── .gitkeep
```

## Deployment Steps

### Step 1: Prepare Repository
1. Ensure all files are committed to your Git repository
2. Make sure the repository is accessible (public or connected private repo)

### Step 2: Create Render Service
1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click "New +" → "Web Service"
3. Connect your Git repository
4. Configure settings as specified above

### Step 3: Configure Build Settings
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: (leave empty, will use Procfile)

### Step 4: Deploy
1. Click "Create Web Service"
2. Render will automatically build and deploy your application
3. The build process will:
   - Install Python 3.11.9
   - Install dependencies from requirements.txt
   - Load pre-trained models
   - Start the Gunicorn server

### Step 5: Verify Deployment
- Access your application at the provided Render URL
- Test image upload functionality
- Camera functionality will be disabled in deployment (as intended)

## Important Notes

### Model Loading
- Pre-trained models are included in the repository
- Models are automatically loaded on application startup
- If model loading fails, the application will not start

### Camera Functionality
- Camera features are automatically disabled in deployment environments
- This is by design as Render servers don't have camera access
- Local development still supports camera functionality

### File Storage
- Uploaded images are stored temporarily in the `uploads/` directory
- Processed results are stored in the `results/` directory
- These directories are created automatically if they don't exist

### Performance Considerations
- Single worker configuration for memory efficiency
- 120-second timeout for model loading and processing
- Headless OpenCV version for server environments

### Troubleshooting

**Build Fails**:
- Check requirements.txt for version conflicts
- Ensure all dependencies are available for Python 3.11.9

**Application Won't Start**:
- Check build logs for model loading errors
- Verify pre-trained model files are present in repository

**Slow Performance**:
- Model loading happens on startup (one-time cost)
- Consider upgrading Render plan for better performance

**Memory Issues**:
- TensorFlow and models require significant memory
- Use Render's Standard plan or higher for production

## Security Considerations

- File upload size limited to 16MB
- Only image files are accepted
- Temporary file storage only
- No persistent user data storage

## Scaling

For high-traffic deployment:
- Consider using Redis for session management
- Implement proper file cleanup routines
- Use cloud storage (AWS S3, Google Cloud) for file handling
- Monitor memory usage and scale accordingly

## Support

For deployment issues:
1. Check Render build logs
2. Monitor application logs in Render dashboard
3. Test locally first to isolate issues
4. Verify all model files are included in deployment
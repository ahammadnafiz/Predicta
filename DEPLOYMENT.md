# Predicta Deployment Guide

## Streamlit Cloud Deployment

### Prerequisites
- GitHub account
- Streamlit Cloud account (sign up at [share.streamlit.io](https://share.streamlit.io))
- Repository must be public or you need Streamlit Cloud Pro

### Step-by-Step Deployment

1. **Fork the Repository**
   - Go to [https://github.com/ahammadnafiz/Predicta](https://github.com/ahammadnafiz/Predicta)
   - Click "Fork" button
   - Fork to your account

2. **Connect to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Grant necessary permissions

3. **Deploy the App**
   - Click "New app"
   - Select your forked repository
   - Set the following:
     - **Repository**: `your-username/Predicta`
     - **Branch**: `main`
     - **Main file path**: `streamlit_app.py`
   - Click "Deploy!"

4. **Wait for Deployment**
   - Streamlit Cloud will install dependencies
   - This may take 5-10 minutes
   - Watch the logs for any errors

### Troubleshooting

#### Access Denied Error
If you see "You do not have access to this app or it does not exist":

1. **Check Repository Permissions**
   - Ensure your repository is public, OR
   - Have Streamlit Cloud Pro for private repos

2. **Verify GitHub Connection**
   - Go to Streamlit Cloud settings
   - Check GitHub integration
   - Re-authorize if needed

3. **Repository URL Issues**
   - Make sure you're using the correct repository URL
   - Repository name should match exactly

#### Deployment Failures

1. **Dependencies Issues**
   - Check `requirements.txt` is properly formatted
   - Ensure all packages are available on PyPI
   - Use version pinning for stability

2. **Import Errors**
   - Verify `streamlit_app.py` is in root directory
   - Check Python path configurations
   - Ensure all modules are properly structured

3. **Memory Issues**
   - Large ML models may exceed memory limits
   - Consider model optimization
   - Use model caching strategies

### Environment Variables
If your app needs environment variables:
1. Go to app settings in Streamlit Cloud
2. Add environment variables in "Secrets" section
3. Use `st.secrets` to access them in your app

### Custom Domain (Pro Feature)
For custom domains:
1. Upgrade to Streamlit Cloud Pro
2. Configure domain in app settings
3. Update DNS records as instructed

## Local Development

### Setup
```bash
git clone https://github.com/your-username/Predicta.git
cd Predicta
pip install -e .
```

### Run Locally
```bash
streamlit run streamlit_app.py
```

### Testing Before Deployment
Always test locally before deploying:
```bash
# Test the exact entry point used by Streamlit Cloud
python streamlit_app.py
```

## Performance Optimization

### For Streamlit Cloud
1. **Optimize Dependencies**
   - Remove unused packages
   - Use specific versions
   - Consider lighter alternatives

2. **Model Optimization**
   - Use model compression
   - Implement lazy loading
   - Cache frequently used models

3. **Data Handling**
   - Limit default data size
   - Implement data sampling
   - Use efficient data structures

### Monitoring
- Monitor app performance in Streamlit Cloud dashboard
- Check memory usage and processing time
- Set up alerts for app downtime

## Support

If you encounter issues:
1. Check Streamlit Community Forum
2. Review Streamlit Cloud documentation
3. Contact the app developer: ahammadnafiz@outlook.com

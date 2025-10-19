FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all dashboard files
COPY . .

# Set Streamlit environment variables
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHERUSAGESTATS=false
ENV STREAMLIT_GLOBAL_DEVELOPMENTMODE=false
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_ENABLEXSFPROTECTION=false

# Expose port
EXPOSE 8501

# Run Streamlit in production mode
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

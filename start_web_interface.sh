#!/bin/bash
# Start the Health Digital Twin Web Interface

echo "=========================================="
echo "Health Digital Twin Prediction Platform"
echo "=========================================="
echo ""
echo "Starting web interface..."
echo ""

cd /home/tc115/Yue/Patient_Digital_Twin_Systems

# Start Streamlit
streamlit run web_app.py \
  --server.port 8501 \
  --server.headless true \
  --server.address 0.0.0.0 \
  --browser.gatherUsageStats false

echo ""
echo "Web interface stopped."

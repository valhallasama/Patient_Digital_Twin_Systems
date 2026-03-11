#!/bin/bash
# Setup automated daily data acquisition using cron

echo "Setting up automated daily data acquisition..."

# Make scripts executable
chmod +x run_daily_data_acquisition.py

# Create logs directory
mkdir -p logs

# Add cron job (runs daily at 2 AM)
SCRIPT_PATH="$(pwd)/run_daily_data_acquisition.py"
CRON_JOB="0 2 * * * cd $(pwd) && /usr/bin/python3 $SCRIPT_PATH >> logs/cron.log 2>&1"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "run_daily_data_acquisition.py"; then
    echo "Cron job already exists"
else
    # Add cron job
    (crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -
    echo "✓ Cron job added: Daily at 2 AM"
fi

echo ""
echo "Setup complete!"
echo ""
echo "The system will now automatically:"
echo "  1. Search public repositories for new health datasets"
echo "  2. Download up to 10GB of data per day"
echo "  3. Validate all downloaded datasets"
echo "  4. Clean and prepare data for training"
echo "  5. Track data sources and lineage"
echo ""
echo "To run manually:"
echo "  python3 run_daily_data_acquisition.py"
echo ""
echo "To view cron jobs:"
echo "  crontab -l"
echo ""
echo "To remove cron job:"
echo "  crontab -e  # then delete the line"
echo ""

#!/bin/bash
echo "🚀 Restarting Disposition Engine and Monitoring..."

# 1. Restart the API
sudo systemctl enable disposition_api
sudo systemctl start disposition_api

# 2. Restart Prometheus
sudo systemctl enable prometheus
sudo systemctl start prometheus

# 3. Restart Grafana
sudo systemctl enable grafana-server
sudo systemctl start grafana-server

echo "✅ All services restarted!"
echo "API: http://65.0.97.13:8080"
echo "Grafana: http://65.0.97.13:3000"

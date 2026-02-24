**Grafana & Prometheus Monitoring for Disposition API**

- **Files added**:
  - `qwen_3b/deployment/monitoring/grafana_dashboard.json` — Grafana dashboard JSON template you can import.

**1) Prometheus: scrape config**

Add this job to your `prometheus.yml` scrape_configs (replace `<HOST>` with your server host or IP):

```yaml
scrape_configs:
  - job_name: 'disposition_model'
    static_configs:
      - targets: ['<HOST>:8080']
```

Prometheus will then scrape `http://<HOST>:8080/metrics` which the app exposes.

**2) Import Grafana dashboard**

Option A — Grafana UI (recommended):
- Open Grafana -> Dashboards -> Manage -> Import
- Upload `qwen_3b/deployment/monitoring/grafana_dashboard.json` or paste its JSON
- When prompted, select the Prometheus data source (or leave default variable `DS_PROMETHEUS` pointing to your Prometheus)

Option B — Grafana HTTP API (automatable):
- Create a Grafana API key (Admin > API Keys) with `Editor` role
- POST the JSON to the dashboards API:

```bash
API_KEY=YOUR_GRAFANA_API_KEY
GRAFANA_URL=http://<grafana-host>:3000
curl -s -X POST -H "Content-Type: application/json" -H "Authorization: Bearer $API_KEY" \
  -d @qwen_3b/deployment/monitoring/grafana_dashboard.json \
  $GRAFANA_URL/api/dashboards/db
```

**3) Dashboard notes & queries**
- GPU utilization: `disposition_gpu_util_percent{gpu="0"}`
- GPU memory used/total: `disposition_gpu_mem_used_mb{gpu="0"}` / `disposition_gpu_mem_total_mb{gpu="0"}`
- Inference latency (P95): `histogram_quantile(0.95, sum(rate(disposition_inference_seconds_bucket[5m])) by (le))`
- Request rate: `rate(disposition_requests_total[1m])`
- Errors rate: `rate(disposition_request_errors_total[1m])`

**4) Using the dashboard**
- Ensure Prometheus is scraping the API (`/metrics`). Check Prometheus target page: `http://<prometheus-host>:9090/targets`
- In Grafana, set the data source to your Prometheus instance if not automatically detected.
- Open the imported dashboard; panels update automatically (refresh set to 10s).

**5) Optional improvements**
- Add per-disposition counters (e.g., `disposition_result{label="ANSWERED"}`) — instrument `model.predict` to increment counters per returned `disposition`. I can add this if you want.
- Add alerts in Grafana or Prometheus Alertmanager (e.g., GPU utilization > 95% for 2m or inference error rate spike).

**6) Troubleshooting**
- If `/metrics` doesn't appear in Prometheus, confirm the server is accessible and firewall allows port 8080.
- If GPU metrics show 0, ensure `nvidia-smi` is present and the driver is installed; the app sets `disposition_gpu_available` to 0 when `nvidia-smi` fails.

If you'd like, I can:
- Add per-disposition Prometheus counters to the server (recommended),
- Provide a ready-to-import Grafana dashboard JSON tailored further (panels, thresholds, colors),
- Or generate a Grafana dashboard provisioning file for automated deployment.

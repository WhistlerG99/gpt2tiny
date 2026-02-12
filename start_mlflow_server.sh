mlflow server   --backend-store-uri /teamspace/studios/this_studio/gpt2tiny/mlruns   --host 0.0.0.0   --port 5000   --workers 1 --allowed-hosts "*" &
# pkill -f "mlflow" || true
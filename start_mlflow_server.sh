mlflow server \
  --host 0.0.0.0 --port 5000 \
  --backend-store-uri file:/teamspace/studios/this_studio/gpt2tiny/mlruns \
  --allowed-hosts "*"
  # --serve-artifacts \
  # --artifacts-destination /teamspace/s3_folders/mlflow-job-artifacts \
  # --workers 4 \

# mlflow server   --backend-store-uri /teamspace/studios/this_studio/gpt2tiny/mlruns --host 0.0.0.0   --port 5000 --artifacts-destination /teamspace/s3_folders/mlflow --workers 1 --allowed-hosts "*" &
# pkill -f "mlflow" || true
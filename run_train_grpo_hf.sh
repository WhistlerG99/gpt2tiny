MLF_DIR=/teamspace/studios/this_studio/gpt2tiny/mlruns

python train_grpo_hf.py \
    --init-model-path ${MLF_DIR}/599992188948983400/1c02a79ac66343b7a5c96e78879077a9/artifacts/checkpoints/best-step=2300-val_loss=2.0833.ckpt \
    --exp-name grpo-hf \
    --run-prefix grpo-hf \
    --model-name GRPOGPT2HF \
    --num-gen 8 \
    --max-seq-len 248 \
    --temperature 0.9 \
    --top-k 50 \
    --top-p 0.95 \
    --lr 3e-5 \
    --warmup-ratio 0.05 \
    --batch-size 2 \
    --max-steps 100 \
    --log-interval 1 \
    --val-interval 5 \
    --val-batches 16 \
    --num-workers 4 \
    --grad-accum-step 8 \
    --gen-max-tokens 248 \
    --gen-temperature 0.9 \
    --gen-top-k 50 \
    --gen-top-p 0.95 \
    --kl-beta 0.03 \
    --clip-eps 0.2
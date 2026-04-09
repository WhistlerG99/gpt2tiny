MLF_DIR=/teamspace/studios/this_studio/gpt2tiny/mlruns

EXP_NAME=grpo-hf
# EXP_NAME=test

RUN_PREFIX=grpo-hf-embedding-reward
# RUN_PREFIX=grpo-hf-embedding-reward-sharpend
# RUN_PREFIX=test


MODEL_NAME=GRPOGPT2HF_embedding_reward
# MODEL_NAME=test

python train_grpo_hf.py \
    --init-model-path ${MLF_DIR}/599992188948983400/1c02a79ac66343b7a5c96e78879077a9/artifacts/checkpoints/best-step=2300-val_loss=2.0833.ckpt \
    --exp-name $EXP_NAME \
    --run-prefix $RUN_PREFIX \
    --model-name $MODEL_NAME \
    --num-gen 8 \
    --max-seq-len 128 \
    --temperature 0.8 \
    --top-k 50 \
    --top-p 0.9 \
    --lr 4e-5 \
    --warmup-ratio 0.05 \
    --batch-size 4 \
    --max-steps 100 \
    --log-interval 1 \
    --val-interval 5 \
    --val-batches 16 \
    --num-workers 4 \
    --grad-accum-step 8 \
    --gen-max-tokens 128 \
    --gen-temperature 0.8 \
    --gen-top-k 50 \
    --gen-top-p 0.9 \
    --kl-beta 0.07 \
    --entropy-beta 0.02 \
    --clip-eps 0.2 \
    --rw-words 0.1 \
    --rw-pos 0.1 \
    --rw-coherence 0.4 \
    --rw-format 0.0 \
    --rw-repetition-penalty 0.2 \
    --rw-min-chars 750 \
    --rw-max-chars 2000 \
    --rw-min-sentences 5 \
    --rw-max-sentences 10
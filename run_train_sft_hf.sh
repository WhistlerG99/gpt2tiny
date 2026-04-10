MLF_DIR=/teamspace/studios/this_studio/gpt2tiny/mlruns

EXP_NAME=sft-hf
# EXP_NAME=test

RUN_PREFIX=sft-hf
# RUN_PREFIX=test

MODEL_NAME=SFTGPT2HF
# MODEL_NAME=test


python train_sft_hf.py \
    --exp-name $EXP_NAME \
    --run-prefix $RUN_PREFIX \
    --model-name $MODEL_NAME \
    --lr 3e-5 \
    --warmup-ratio 0.05 \
    --batch-size 8 \
    --max-steps 100 \
    --log-interval 10 \
    --val-interval 100 \
    --val-batches 200 \
    --num-workers 4 \
    --grad-accum-step 8 \
    --gen-max-tokens 248
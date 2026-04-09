MLF_DIR=/teamspace/studios/this_studio/gpt2tiny/mlruns

EXP_NAME=tinystories-pretrain
# EXP_NAME=test

RUN_PREFIX=pretrain
# RUN_PREFIX=test

MODEL_NAME=tinystories-pretrain
# MODEL_NAME=test

TOKENIZER_PATH=/teamspace/studios/this_studio/gpt2tiny/data/tok4096_tinystories.model


python pretrain.py \
    --exp-name $EXP_NAME \
    --run-prefix $RUN_PREFIX \
    --model-name $MODEL_NAME \
    --tokenizer-path $TOKENIZER_PATH \
    --block-size 512 \
    --n-layers 8 \
    --n-heads 8 \
    --n-embed 512 \
    --dropout 0.2 \
    --max-steps 20000 \
    --lr 3e-5 \
    --warmup-ratio 0.05 \
    --batch-size 64 \
    --log-interval 25 \
    --val-interval 25 \
    --val-batches 200 \
    --num-workers 4 \
    --grad-accum-step 4 \
    --gen-max-tokens 128 \
    --gen-temperature 0.8 \
    --gen-top-k 50 \
    --gen-top-p 0.9
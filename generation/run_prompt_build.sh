IDX=02

PROMPT_BASE_DIR=/teamspace/studios/this_studio/gpt2tiny
PROMPT_DIR=${PROMPT_BASE_DIR}/data/TinyStories_custom_prompts_sft_v1
COMP_DIR=./batch_data

python build_prompts.py \
    --prompts ${PROMPT_DIR}/data${IDX}.json \
    --output ${COMP_DIR}/${IDX}/prompts${IDX}.jsonl \
    --warnings-file ${COMP_DIR}/${IDX}/prompt_build_warnings.txt
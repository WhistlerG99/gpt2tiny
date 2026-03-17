IDX=01

PROMPT_BASE_DIR=/teamspace/studios/this_studio/gpt2tiny
PROMPT_DIR=${PROMPT_BASE_DIR}/data/TinyStories_custom_prompts_sft_v1
COMP_DIR=./batch_data

python build_validation.py \
    --prompts ${PROMPT_DIR}/data${IDX}.json \
    --completions ${COMP_DIR}/${IDX}/completions${IDX}.jsonl \
    --output ${COMP_DIR}/${IDX}/validation_prompts${IDX}.jsonl \
    --model gpt-5.4 \
    --max-output-tokens 248 \
    --warnings-file ${COMP_DIR}/${IDX}/validation_build_warnings.txt
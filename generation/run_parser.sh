# python parse_validation_results.py \
#   --prompts ../gpt2tiny/data/TinyStories_custom_prompts_sft_v1/data00.json \
#   --completions ./batch_prompts_test/completions00.jsonl \
#   --validations ./batch_prompts_test/validation_response00.jsonl \
#   --output-prefix batch_prompts_test/validation_parsed00

IDX=01

PROMPT_BASE_DIR=/teamspace/studios/this_studio/gpt2tiny
PROMPT_DIR=${PROMPT_BASE_DIR}/data/TinyStories_custom_prompts_sft_v1
COMP_DIR=./batch_data

python parse_results.py \
    --prompts ${PROMPT_DIR}/data${IDX}.json \
    --completions ${COMP_DIR}/${IDX}/completions${IDX}.jsonl \
    --validations ${COMP_DIR}/${IDX}/validations${IDX}.jsonl \
    --passed-results ${COMP_DIR}/${IDX}/data${IDX}.json \
    --failed-results ${COMP_DIR}/${IDX}/failed${IDX}.json \
    --summary ${COMP_DIR}/${IDX}/summary${IDX}.json
    
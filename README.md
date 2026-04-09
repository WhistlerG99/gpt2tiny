# tinyGPT2

Code for performing pretraining, supervised finetuning, and RLHF using GRPO on a small GPT2 type model. The code for the model is based on [smolGPT](https://github.com/Om-Alve/smolGPT).

Currently, the code is setup to do the SFT and RLHF on the Huggingface implementation of [GPT2](https://huggingface.co/openai-community/gpt2) 

---
## Installation

```bash
pip install -r requirements.txt
pip install .
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+ with CUDA

## Dataset

Data for pretraining is done with the [Tinystories dataset](https://huggingface.co/datasets/roneneldan/TinyStories). 

Data for SFT and RLHF was created by programmically generating story prompts and generating completions using a teacher model like GPT5. The code for generating the prompts and formatting them for batch jobs in OpenAI is in [generation/](generation/).

My intention was to refine the pretrained model with SFT and RLHF, but the model has too limited a vocabulary for that, so presently I'm using the Huggingface implementation of [GPT2](https://huggingface.co/openai-community/gpt2) for those parts. 

## Training

1. **Create Dataset**

    Currently, you need to specify the path to the dataset you are using in [preprocess.py](preprocess.py)

- **Option 1: Pretraining**
```bash
python preprocess.py prepare-dataset --vocab-size 4096
```
- **Option 2: Supervised Finetuning**
```bash
python preprocess.py pretokenize-sft
```

- **Option 3: RLHF**
```bash
python preprocess.py pretokenize-rlhf
```

2. **Training**
- **Option 1: Pretraining**
```bash
./run_pretrain.sh
```
- **Option 2: Supervised Finetuning**
```bash
./run_train_sft_hf.sh
```

- **Option 3: RLHF**
```bash
./run_train_grpo_hf.sh
```


### Training RIG SPECS (Rented via LightningAI)  
- **GPU**: NVIDIA T4 Graphics Card (Optimized for AI workloads)  
- **CPUs**: 4
- **RAM**: 16 GB  
- **VRAM**: 16 GB  
---

# llm-pretraining-marathi
Repository to pre-train an LLM model for Marathi literature. Marathi is my mother tongue and hasn't been widely explored language when it comes to LLM applications.

marathi-llm/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── prepare_data.py
│   │   └── download_datasets.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── pretraining.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── scripts/
│   ├── download_data.sh
│   ├── prepare_data.sh
│   └── train_model.sh
├── configs/
│   ├── data_config.json
│   └── training_config.json
├── data/
│   ├── raw/
│   └── processed/
└── models/
    └── marathi-llm/


# Marathi Literature sources

Here are some link to Marathi literaure sources

OSCAR Corpus - Contains a substantial amount of Marathi text crawled from the web

Available at: https://oscar-corpus.com/


AI4Bharat IndicCorp - A large-scale Indic languages corpus

Contains over 8.9 GB of Marathi text
Available at: https://indicnlp.ai4bharat.org/corpora/


Marathi Wikisource - Classic Marathi literature

https://dumps.wikimedia.org/mrwikisource/


IITB Marathi Corpus - A parallel corpus with English translations

http://www.cfilt.iitb.ac.in/iitb_parallel/


Project Gutenberg - Some classic Marathi texts

https://www.gutenberg.org/browse/languages/mr


Maharashtra State Archives - Historical documents and literature

https://maharashtracivilservice.gov.in/en


Sahitya Akademi - Indian literature academy with Marathi works

http://sahitya-akademi.gov.in/

# Step by step guide

## Download 
mkdir -p data/raw
python src/data/download_datasets.py --output_dir data/raw

## Proces
python src/data/prepare_data.py --input_dir data/raw --output_dir data/processed --train_split 0.9 --val_split 0.1

## Train a custom tokenizer
python src/models/pretraining.py --data_dir data/processed --output_dir models/marathi-llm --train_tokenizer --vocab_size 50000

## Pre-train the model

python src/models/pretraining.py \
    --data_dir data/processed \
    --output_dir models/marathi-llm \
    --model_name_or_path gpt2 \
    --max_steps 100000 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --warmup_steps 1000 \
    --fp16


For larger models, you can start from a multilingual foundation:

python src/models/pretraining.py \
    --data_dir data/processed \
    --output_dir models/marathi-llm-large \
    --model_name_or_path google/mt5-small \
    --max_steps 100000 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 3e-5 \
    --fp16

## Step 5 evauate the model

python evaluate.py

## Advance techniques

Distributed Training: Use PyTorch's DistributedDataParallel for multi-GPU training
Gradient Checkpointing: To handle larger models with limited memory:

```
model.gradient_checkpointing_enable()
```
Mixed Precision Training: Use FP16 to speed up training and reduce memory usage (already included in the training script with --fp16)
Curriculum Learning: Start with shorter sequences and gradually increase length
Efficient Architectures: Consider using models like LLaMA, Falcon, or GPT-Neo architectures which have optimized implementations


## What compute resource are required

Compute Resources: Pre-training requires significant GPU resources. Consider:

Using cloud providers like AWS, GCP, or Azure with V100/A100 GPUs

Training on multiple smaller GPUs using distributed training

Using gradient checkpointing and mixed precision to reduce memory requirements

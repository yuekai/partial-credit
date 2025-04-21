# install requirements

```shell
conda create -n minitrain python=3.12 -y; conda activate minitrain
pip install torch
pip install -r requirements.txt
```

# Process Input Data

You first need to tokenize your data and get the input tokens and the output label tokens. The resulting label tokens will contain a mask token (`-100`) for the `user` and `system` roles, and unmask the tokens for the `assistant` role content. If you wish to pretrain (i.e. train on all tokens in the input) include the data in a single `pretrain` role message.

```shell
TOKENIZERS_PARALLELISM=False \
python process_data.py \
--input-file "data_with_messages_format.jsonl" \
--output-file "tokenized_data.jsonl" \
--model-name-or-path "Qwen/Qwen2.5-1.5B-instruct" \
--max-sample-num-tokens 32768
```

## assumptions on the data
The data must be a `json list` format (each line a json), and each sample should have a `messages` field formatted like this:


```json
{"messages": [{"role": "user","content": "<input 1>"},{"role": "assistant","content": "<output 1>"}]}
{"messages": [{"role": "user","content": "<input 2>"},{"role": "assistant","content": "<output 2>"}]}
{"messages": [{"role": "pretrain","content": "<pretrain data>"}]} #this sample will have completely unmasked tokens in the labels.
```

the data processing script will populate `input_ids`, `labels` and the sample length in the `len` keys. To do so the data processor uses the chat template included in the tokenizer, make sure the tokenizer has a proper chat template set up.

NOTE: check the printed examples at the end and make sure the samples look properly formatted and that the masked part of the labels correspods to anything that the model would not learn to generate (although it would still condition the model -- the $x$ in $p(y|x)$ ).

## Pretraining
if you want to pretrain on some samples, such samples should have a messages format with a single element with the role `pretrain` and the data in the `content` key. This would use void using the chat template and the complete string in the `content` key would be unmasked.

## launch training

all training parameters can be found in [train.py](./train.py). Make sure to use the tokenized data created above as the input here.

```shell
torchrun --nnodes=1 --nproc-per-node=8 train.py \
        --output-dir ./experiment_checkpoints_loggin_etc/ \
        --data-path ./tokenized_data.jsonl \
        --model-name-or-path Qwen/Qwen2.5-1.5B-instruct \
        --min-samples-per-checkpoint 10000 \
        --num-warmup-steps 20 \
        --max-tokens-per-gpu \
        --batch-size 128 \
        --use-liger-kernels \
        --seed 893 \
        --fsdp-sharding-strategy FULL_SHARD \
        --learning-rate 6e-6
```
# install requirements

```shell
pip install -r requirements.txt
```

# Process Input Data

```shell
python process_data.py --input-file "deepscaler_r1_qwen1.5b_debug.jsonl" --output-file "test.jsonl" --model-name-or-path "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
```

## assumptions on the data
The data must be a json list format (each line a json), and each sample should have a `messages` format like this:

```json
{
  #... other fields that are unused
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful AI assistant."
    },
    {
      "role": "user",
      "content": "What is the capital of France?"
    },
    {
      "role": "assistant",
      "content": "The capital of France is Paris."
    }
  ]
}
```

the data processing script will populate `input_ids`, `labels` and the sample length in the `len` keys. To do so the data processor uses the chat template included in the tokenizer, make sure the tokenizer has a proper chat template set up.

NOTE: check the printed examples at the end and make sure the samples look properly formatted and that the masked part of the labels correspods to anything that the model would not learn to generate (although it would still condition the model -- the $x$ in $p(y|x)$ ).

## Pretraining
if you want to pretrain on some samples, such samples should have a messages format with a single element with the role `pretrain` and the data in the `content` key. This would use void using the chat template and the complete string in the `content` key would be unmasked.

# launch training

all training parameters can be found in [train.py](./train.py)
```shell
torchrun --nnodes=1 --nproc-per-node=8 train.py     --output-dir /tmp/my_training_output     --min-samples-per-checkpoint 10000 --max-tokens-per-gpu 10000 --use-liger-kernels
```
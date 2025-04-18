# install requirements

```shell
pip install -r requirements.txt
```


# Process Input Data

```shell
python process_data.py --input-file "deepscaler_r1_qwen1.5b_debug.jsonl" --output-file "test.jsonl" --model-name-or-path "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
```

# launch training

all training parameters can be found in [train.py](./train.py)
```shell
torchrun --nnodes=1 --nproc-per-node=8 train.py     --output-dir /tmp/my_training_output     --min-samples-per-checkpoint 10000 --max-tokens-per-gpu 10000 --use-liger-kernels
```
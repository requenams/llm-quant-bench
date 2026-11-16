# How much does quantization really cost?

A small, reproducible benchmark of post training quantization for an open LLM.
It compares the same model in 16 bit, 8 bit and 4 bit precision and asks two questions:

1. **What do we gain?** Memory footprint, decode throughput and prefill throughput.
2. **What do we lose, and is the loss real?** Perplexity and MMLU accuracy, with
   paired bootstrap confidence intervals and an exact McNemar test, so a small drop
   in accuracy is not mistaken for a real one (or the other way round).

Most quick quantization comparisons report a single accuracy number per variant.
With a few hundred questions, differences of one or two points are often within noise,
so this project evaluates every variant on exactly the same questions and text windows
and compares them pairwise.

## Variants

| Variant | Method | Measured |
|---|---|---|
| `fp16` | 16 bit floats (baseline) | accuracy, memory, speed |
| `int8` | bitsandbytes LLM.int8 | accuracy, memory, speed |
| `nf4` | bitsandbytes 4 bit NormalFloat | accuracy, memory, speed |
| `fp4` | bitsandbytes 4 bit float | accuracy, memory, speed |
| `mo_int8_sq` | NVIDIA Model Optimizer, INT8 SmoothQuant | accuracy only (simulated) |
| `mo_int4_awq` | NVIDIA Model Optimizer, INT4 AWQ | accuracy only (simulated) |

The Model Optimizer variants use simulated quantization in PyTorch (quantize and
dequantize), which gives a realistic estimate of accuracy. Their speed and memory
benefits only appear once the model is exported to an inference engine such as
TensorRT-LLM, so those numbers are not reported here.

## Metrics

- **Memory**: model footprint and peak GPU memory during generation.
- **Speed**: decode throughput (batch size 1, greedy, 128 new tokens) and prefill
  throughput (one forward pass over 1024 tokens), averaged over 5 runs after warmup.
- **Perplexity** on the WikiText-2 test set, split into non overlapping windows of
  1024 tokens. Keeping the loss of each window lets us bootstrap the perplexity ratio.
- **MMLU** zero shot accuracy on a fixed random subset of 1000 test questions, scored
  by comparing the logits of the answer letters.

## Statistics

- Accuracy change: paired bootstrap over questions (10,000 resamples, 95% CI).
- Exact McNemar test on the questions where baseline and quantized model disagree.
- Perplexity change: paired bootstrap over text windows.

## Run it (Google Colab, free T4 GPU)

```bash
pip install -r requirements.txt
python run_benchmark.py --variant fp16
python run_benchmark.py --variant int8
python run_benchmark.py --variant nf4
python run_benchmark.py --variant fp4
python analyze.py --baseline fp16
```

Optional, with `pip install nvidia-modelopt`:

```bash
python run_benchmark.py --variant mo_int8_sq
python run_benchmark.py --variant mo_int4_awq
```

Default model: `Qwen/Qwen2.5-1.5B` (change it with `--model`). Each run saves a JSON
file with the raw, per item results in `results/`, and `analyze.py` writes
`results/summary.md` plus two plots: accuracy versus memory and accuracy versus speed.

## Results

*To be filled in after running the benchmark.*

## Limitations

- Speed is measured with Hugging Face `generate` at batch size 1. Production engines
  (TensorRT-LLM, vLLM) and larger batches can change the picture a lot, especially
  for bitsandbytes, whose kernels favour memory savings over speed.
- A T4 has no native support for FP8 or bfloat16, so those formats are not compared.
- 1000 MMLU questions keep runs short on a free GPU; the confidence intervals show
  how much that matters.

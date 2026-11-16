"""Load the same model in different numerical precisions."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Variants measured for real (memory and speed are meaningful).
BNB_VARIANTS = {
    "fp32": "full precision, 32 bit floats",
    "fp16": "16 bit floats (baseline on GPU)",
    "bf16": "16 bit bfloat",
    "int8": "bitsandbytes LLM.int8",
    "nf4": "bitsandbytes 4 bit NormalFloat",
    "fp4": "bitsandbytes 4 bit float",
}

# Simulated quantization with NVIDIA Model Optimizer: weights and activations are
# quantized and dequantized in PyTorch, so accuracy is realistic but memory and
# speed are not (real speedups need a deployment engine such as TensorRT-LLM).
MODELOPT_VARIANTS = {
    "mo_int8_sq": "ModelOpt INT8 SmoothQuant (simulated)",
    "mo_int4_awq": "ModelOpt INT4 AWQ (simulated)",
}

ALL_VARIANTS = {**BNB_VARIANTS, **MODELOPT_VARIANTS}


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_float(model_id, dtype, device):
    return AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=device)


def load_model(model_id, variant, device=None, calib_texts=None):
    """Return (model, tokenizer, is_simulated)."""
    if variant not in ALL_VARIANTS:
        raise ValueError(f"Unknown variant {variant}. Options: {list(ALL_VARIANTS)}")
    device = device or default_device()
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    simulated = False
    if variant in ("fp32", "fp16", "bf16"):
        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[variant]
        model = _load_float(model_id, dtype, device)
    elif variant == "int8":
        qc = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=qc, device_map=device)
    elif variant in ("nf4", "fp4"):
        qc = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=variant,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=qc, device_map=device)
    else:
        model = _load_float(model_id, torch.float16, device)
        model = _modelopt_quantize(model, tok, variant, calib_texts, device)
        simulated = True

    model.eval()
    return model, tok, simulated


def _modelopt_quantize(model, tok, variant, calib_texts, device, n_calib=64, seq_len=512):
    """Post training quantization with NVIDIA Model Optimizer (pip install nvidia-modelopt)."""
    import modelopt.torch.quantization as mtq

    cfg = {"mo_int8_sq": mtq.INT8_SMOOTHQUANT_CFG, "mo_int4_awq": mtq.INT4_AWQ_CFG}[variant]
    if not calib_texts:
        raise ValueError("ModelOpt variants need calibration texts")
    batches = [
        tok(t, return_tensors="pt", truncation=True, max_length=seq_len).input_ids.to(device)
        for t in calib_texts[:n_calib]
    ]

    def forward_loop(m):
        with torch.no_grad():
            for ids in batches:
                m(ids)

    return mtq.quantize(model, cfg, forward_loop)


def memory_footprint_gb(model):
    return model.get_memory_footprint() / 1024**3

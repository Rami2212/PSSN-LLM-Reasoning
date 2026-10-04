"""Environment inspection and model-loading helpers for Colab experiments.

The module deliberately does not import PyTorch or Transformers at import time.
This keeps environment diagnostics and unit tests useful before the relatively
large research dependencies have been installed.
"""

from __future__ import annotations

import importlib
import json
import logging
import platform
from typing import Any

DEFAULT_MODEL_ID = "Qwen/Qwen3-4B"
LOGGER = logging.getLogger(__name__)


def _import_dependency(name: str) -> Any:
    """Import a dependency with an actionable installation error."""
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise RuntimeError(
            f"Missing dependency '{name}'. Install the project requirements first: "
            "python -m pip install -r requirements.txt"
        ) from exc


def select_inference_dtype(torch_module: Any | None = None) -> tuple[Any, str]:
    """Select BF16 when supported, otherwise FP16 on CUDA or FP32 on CPU.

    FP32 is retained as a diagnostic fallback for non-Colab development. The
    4B model is intended to run on a CUDA runtime, where this function always
    chooses BF16 or FP16 and never enables quantization.
    """
    torch = torch_module or _import_dependency("torch")
    if not torch.cuda.is_available():
        return torch.float32, "float32"

    bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    if bf16_supported:
        return torch.bfloat16, "bfloat16"
    return torch.float16, "float16"


def collect_environment_info(torch_module: Any | None = None) -> dict[str, Any]:
    """Return JSON-serializable Python, CUDA, and GPU metadata."""
    torch = torch_module or _import_dependency("torch")
    cuda_available = bool(torch.cuda.is_available())
    dtype, dtype_name = select_inference_dtype(torch)
    del dtype  # Only the portable name belongs in serialized experiment data.

    devices: list[dict[str, Any]] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "compute_capability": (
                        f"{properties.major}.{properties.minor}"
                    ),
                    "total_memory_gib": round(
                        properties.total_memory / (1024**3), 2
                    ),
                }
            )

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "cuda_available": cuda_available,
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
        "selected_dtype": dtype_name,
        "gpu_count": len(devices),
        "gpus": devices,
    }


def log_environment_info(
    torch_module: Any | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Collect and log environment metadata, returning the same dictionary."""
    info = collect_environment_info(torch_module)
    (logger or LOGGER).info("Runtime environment:\n%s", json.dumps(info, indent=2))
    return info


def load_qwen_model(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    require_cuda: bool = True,
    torch_module: Any | None = None,
    transformers_module: Any | None = None,
) -> tuple[Any, Any, str]:
    """Load the Qwen tokenizer and unquantized causal LM for inference.

    Returns ``(tokenizer, model, dtype_name)``. ``device_map='auto'`` lets
    Accelerate place the model on the available Colab GPU without hard-coding a
    device index.
    """
    torch = torch_module or _import_dependency("torch")
    transformers = transformers_module or _import_dependency("transformers")

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. In Colab choose Runtime > Change runtime "
            "type > Hardware accelerator > GPU, then run the notebook again."
        )

    dtype, dtype_name = select_inference_dtype(torch)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tokenizer, model, dtype_name


def generate_validation_response(
    tokenizer: Any,
    model: Any,
    *,
    prompt: str = "What is 17 + 25? Give a short explanation and the answer.",
    max_new_tokens: int = 96,
) -> str:
    """Run one deterministic chat generation and return only new model text."""
    messages = [{"role": "user", "content": prompt}]
    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=True,
    )
    model_inputs = model_inputs.to(model.device)

    torch = _import_dependency("torch")
    with torch.inference_mode():
        output_ids = model.generate(
            model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0, model_inputs.shape[-1] :]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if not response:
        raise RuntimeError("The model completed generation without returning text.")
    return response


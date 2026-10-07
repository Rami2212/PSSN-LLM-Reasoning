"""Reusable Qwen3 inference wrapper with reproducible token accounting."""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass, field
from typing import Any

from src.evaluation.metrics import InferenceMeasurement, collect_model_metadata
from src.utils.environment import DEFAULT_MODEL_ID, load_qwen_model


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    """Serializable generation parameters for baseline experiments."""

    max_new_tokens: int = 1024
    do_sample: bool = False
    temperature: float = 0.6
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        if isinstance(self.max_new_tokens, bool) or self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer.")
        if self.temperature <= 0:
            raise ValueError("temperature must be greater than zero.")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in the interval (0, 1].")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be greater than zero.")

    def model_kwargs(self) -> dict[str, Any]:
        """Return only arguments understood by ``model.generate``."""
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "repetition_penalty": self.repetition_penalty,
        }
        if self.do_sample:
            kwargs.update(temperature=self.temperature, top_p=self.top_p)
        return kwargs

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """One complete model generation and its reproducibility metadata."""

    text: str
    input_tokens: int
    output_tokens: int
    generation_config: dict[str, Any]
    metrics: dict[str, int | float] = field(default_factory=dict)
    model_metadata: dict[str, Any] = field(default_factory=dict)
    finish_reason: str = "unknown"


class QwenGenerator:
    """Generate complete Qwen3 reasoning responses from fixed user prompts."""

    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        *,
        settings: GenerationSettings | None = None,
        model_id: str = DEFAULT_MODEL_ID,
        precision: str | None = None,
        torch_module: Any | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.settings = settings or GenerationSettings()
        self.model_id = model_id
        self.precision = precision
        self._torch = torch_module

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        settings: GenerationSettings | None = None,
        require_cuda: bool = True,
    ) -> "QwenGenerator":
        """Load an unquantized Qwen model using the shared environment policy."""
        tokenizer, model, precision = load_qwen_model(
            model_id=model_id,
            require_cuda=require_cuda,
        )
        return cls(
            tokenizer,
            model,
            settings=settings,
            model_id=model_id,
            precision=precision,
        )

    def _get_torch(self) -> Any:
        if self._torch is None:
            try:
                self._torch = importlib.import_module("torch")
            except ImportError as exc:
                raise RuntimeError(
                    "Missing dependency 'torch'. Install requirements.txt first."
                ) from exc
        return self._torch

    def _seed(self, torch: Any) -> None:
        torch.manual_seed(self.settings.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.settings.seed)

    def generate(self, prompt: str) -> GenerationResult:
        """Generate a full reasoning trace for one prompt."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        torch = self._get_torch()
        with InferenceMeasurement(torch) as measurement:
            self._seed(torch)
            rendered_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            model_inputs = self.tokenizer(
                [rendered_prompt],
                return_tensors="pt",
                add_special_tokens=False,
            )
            if hasattr(model_inputs, "to"):
                model_inputs = model_inputs.to(self.model.device)
            else:
                model_inputs = {
                    name: value.to(self.model.device)
                    for name, value in model_inputs.items()
                }

            input_tokens = int(model_inputs["input_ids"].shape[-1])
            generation_kwargs = self.settings.model_kwargs()
            generation_kwargs["pad_token_id"] = self.tokenizer.eos_token_id
            with torch.inference_mode():
                output_ids = self.model.generate(**model_inputs, **generation_kwargs)

            generated_ids = output_ids[0][input_tokens:]
            output_tokens = len(generated_ids)
            eos_ids = getattr(
                getattr(self.model, "generation_config", None), "eos_token_id", None
            )
            if not isinstance(eos_ids, (int, list, tuple)):
                eos_ids = self.tokenizer.eos_token_id
            if isinstance(eos_ids, int):
                eos_ids = [eos_ids]
            ended_with_eos = bool(output_tokens) and int(generated_ids[-1]) in (eos_ids or [])
            finish_reason = (
                "length" if output_tokens >= self.settings.max_new_tokens
                and not ended_with_eos else "eos" if ended_with_eos else "unknown"
            )
            text = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            ).strip()
            if not text:
                raise RuntimeError("Qwen completed generation without returning text.")

        config = self.settings.to_dict()
        config.update(
            {
                "model_id": self.model_id,
                "precision": self.precision,
                "thinking_enabled": True,
            }
        )
        return GenerationResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            generation_config=config,
            finish_reason=finish_reason,
            metrics=measurement.to_dict(),
            model_metadata=collect_model_metadata(
                self.model,
                torch,
                model_id=self.model_id,
                precision=self.precision,
            ),
        )


"""wenyan (文言) — per-tokenizer 中文 prompt-compression Skill for DeepSeek/Qwen/GLM.

The core primitive (mvp_plan §2) is the *per-tokenizer compression-strategy mapping*:
profile each 国产模型 tokenizer's Chinese subword behavior, record the net-token
outcome of each strategy, and pick the winner per model.

m1 (this release) implements the kill-gate only:
  * the per-tokenizer variance profiler across DeepSeek/Qwen/GLM (assert >5%), and
  * the 助词 (particle-stripping) prototype (assert net-token positive per model).
Full strategy suite + per-model picker + regression harness land in m2/m3.
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = ["__version__"]

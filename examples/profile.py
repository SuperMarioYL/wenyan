# examples/profile.py — 3-line programmatic use of wenyan
from wenyan.profiler import DEFAULT_PROMPT_SUITE, load_model_specs, profile, mean_variance
results = profile(DEFAULT_PROMPT_SUITE, load_model_specs())
print("mean per-tokenizer variance:", round(mean_variance(results, len(DEFAULT_PROMPT_SUITE)), 1), "%")

"""Secret loading for the free stack (HF Hub token, wandb key).

Order: env var -> Kaggle Secrets -> Colab userdata -> .env file. Never logged, never committed.
"""

# TODO: get_secret(name) -> resolve from env / Kaggle / Colab / .env in that order.
# TODO: hf_token() / wandb_key() -> thin wrappers over get_secret with the canonical names.

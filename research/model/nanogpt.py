"""nanoGPT: a minimal GPT, sized by config for proxy (1-3M) or 124M.

Kept deliberately close to Karpathy's nanoGPT so the failure modes match the literature.
"""

# TODO: GPTConfig dataclass -> n_layer, n_head, n_embd, block_size, vocab_size, dropout, bias.
# TODO: GPT(nn.Module) -> embeddings, transformer blocks, lm_head; forward(idx, targets) -> logits, loss.
# TODO: GPT.configure_optimizers(cfg) -> AdamW with param-group weight decay (matches trunk/fork).

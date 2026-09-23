"""
nanogpt_baseline.py

A minimal char-level Transformer (same architecture family as Karpathy's
nanoGPT/minGPT: token+position embeddings -> stacked pre-norm causal
self-attention + MLP blocks -> linear head), trained with plain AdamW
cross-entropy on the SAME tiny-Shakespeare data split used in
language_benchmark.py's scaling-law test:
  - training data: a prefix of the first 90% of tinyshakespeare.txt
  - held-out validation: the SAME fixed 20,000-character slice starting
    at the 90% mark, never seen during training

This exists purely to give an honest, apples-to-apples reference point:
how does a real (if tiny) gradient-trained transformer's held-out
accuracy/bits-per-character and wall-clock training time compare to the
closed-form renormalization pipeline's numbers on identical data?
"""
import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)


def load_data(val_slice=20_000):
    with open("tinyshakespeare.txt", "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}

    n = len(text)
    val_start = int(0.9 * n)
    train_text = text[:val_start]
    val_text = text[val_start:val_start + val_slice]

    encode = lambda s: [stoi[c] for c in s]
    train_data = torch.tensor(encode(train_text), dtype=torch.long)
    val_data = torch.tensor(encode(val_text), dtype=torch.long)
    return train_data, val_data, vocab_size


class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, head_size, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, block_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = self.blocks(tok_emb + pos_emb)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def evaluate(model, data, block_size, batch_size, iters):
    model.eval()
    losses, correct, total = [], 0, 0
    for _ in range(iters):
        x, y = get_batch(data, block_size, batch_size)
        logits, loss = model(x, y)
        losses.append(loss.item())
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += y.numel()
    model.train()
    mean_loss = sum(losses) / len(losses)
    return mean_loss, mean_loss / math.log(2), correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max_iters", type=int, default=3000)
    p.add_argument("--eval_interval", type=int, default=300)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    args = p.parse_args()

    train_data, val_data, vocab_size = load_data()
    print(f"Train chars: {len(train_data)} | Held-out chars: {len(val_data)} | vocab: {vocab_size}")

    model = TinyGPT(vocab_size, args.n_embd, args.n_head, args.n_layer, args.block_size)
    n_params = sum(pp.numel() for pp in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"\nTraining for {args.max_iters} iterations "
          f"(block_size={args.block_size}, batch_size={args.batch_size})...\n")
    t0 = time.perf_counter()
    for it in range(args.max_iters + 1):
        if it % args.eval_interval == 0 or it == args.max_iters:
            tr_loss, tr_bpc, tr_acc = evaluate(model, train_data, args.block_size, args.batch_size, args.eval_iters)
            va_loss, va_bpc, va_acc = evaluate(model, val_data, args.block_size, args.batch_size, args.eval_iters)
            elapsed = time.perf_counter() - t0
            print(f"iter {it:5d} | train bpc {tr_bpc:.3f} acc {tr_acc:.3f} | "
                  f"val bpc {va_bpc:.3f} acc {va_acc:.3f} | {elapsed:6.1f}s")
        xb, yb = get_batch(train_data, args.block_size, args.batch_size)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    total_time = time.perf_counter() - t0
    print(f"\nTotal training wall-clock time: {total_time:.1f}s for {args.max_iters} "
          f"iterations, {n_params:,} parameters")


if __name__ == "__main__":
    main()

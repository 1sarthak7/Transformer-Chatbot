"""
Transformer Chatbot v3 — Trained on 50K QA pairs with GPU acceleration.

Upgrades:
  - Real QA dataset (question -> answer pairs)
  - MPS GPU acceleration (Apple M4)
  - Vocab capped at 10K most common words
  - Progress bar with time estimates
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import time
from collections import Counter

# Import hand-crafted conversations for diversity
from corpus import CONVERSATIONS as CONVERSATIONS_SMALL


# ── Device Selection ───────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Tokenizer ─────────────────────────────────────────────────────

class Tokenizer:
    SPECIAL_TOKENS = ['<pad>', '<unk>', '<user>', '<bot>', '<end>']

    def __init__(self, max_vocab=10000):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.max_vocab = max_vocab

    def build_vocab(self, text):
        words = text.lower().replace('.', '').replace(',', '').replace('?', '').replace('!', '').split()
        self.word_to_id = {tok: i for i, tok in enumerate(self.SPECIAL_TOKENS)}
        idx = len(self.SPECIAL_TOKENS)

        word_counts = Counter(words)
        # Keep only top N most common words (cap vocabulary)
        for word, _ in word_counts.most_common(self.max_vocab - len(self.SPECIAL_TOKENS)):
            if word not in self.word_to_id:
                self.word_to_id[word] = idx
                idx += 1

        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.vocab_size = len(self.word_to_id)

    def encode(self, text):
        words = text.lower().replace('.', '').replace(',', '').replace('?', '').replace('!', '').split()
        return [self.word_to_id.get(w, self.word_to_id['<unk>']) for w in words]

    def decode(self, ids):
        return ' '.join(self.id_to_word.get(i, '<unk>') for i in ids)

    def decode_one(self, token_id):
        return self.id_to_word.get(token_id, '<unk>')


# ── Transformer Components ────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_len=512, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer('mask', mask)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale
        attn = attn.masked_fill(self.mask[:T, :T], float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=4,
                 ff_dim=512, max_len=128, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_encoding = PositionalEncoding(embed_dim, max_len)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)
        self.output.weight = self.token_embedding.weight

    def forward(self, x):
        x = self.token_embedding(x)
        x = self.position_encoding(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        return self.output(x)


# ── Data Preparation ──────────────────────────────────────────────

def create_training_data(token_ids, seq_length=64):
    inputs, targets = [], []
    stride = seq_length // 2
    for i in range(0, len(token_ids) - seq_length, stride):
        chunk = token_ids[i : i + seq_length + 1]
        if len(chunk) < seq_length + 1:
            break
        inputs.append(chunk[:-1])
        targets.append(chunk[1:])
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


# ── Training ──────────────────────────────────────────────────────

def train_model(model, inputs, targets, device, epochs=30, lr=0.0003, batch_size=64):
    model.to(device)
    inputs = inputs.to(device)
    targets = targets.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    warmup_steps = min(500, len(inputs) // batch_size * 3)
    total_steps = (len(inputs) // batch_size) * epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    dataset_size = len(inputs)

    print(f"\n  Dataset size: {dataset_size:,} sequences")
    print(f"  Batch size:   {batch_size}")
    print(f"  Batches/epoch: {dataset_size // batch_size}")
    print(f"  Total steps:  {total_steps:,}")
    print()

    global_step = 0
    epoch_times = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        num_batches = 0
        epoch_start = time.time()

        indices = torch.randperm(dataset_size, device=device)
        for start in range(0, dataset_size, batch_size):
            batch_idx = indices[start:start + batch_size]
            batch_in = inputs[batch_idx]
            batch_tgt = targets[batch_idx]

            logits = model(batch_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), batch_tgt.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_loss = epoch_loss / num_batches
        avg_epoch = sum(epoch_times) / len(epoch_times)
        remaining = avg_epoch * (epochs - epoch)

        # Calculate accuracy every few epochs
        acc_str = ""
        if epoch % 5 == 0 or epoch <= 2 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                sample_idx = torch.randperm(dataset_size, device=device)[:min(500, dataset_size)]
                sample_logits = model(inputs[sample_idx])
                sample_tgt = targets[sample_idx]
                mask = sample_tgt != 0
                preds = sample_logits.argmax(dim=-1)
                correct = ((preds == sample_tgt) & mask).sum().item()
                total = mask.sum().item()
                accuracy = correct / total * 100 if total > 0 else 0
            acc_str = f"  |  Acc: {accuracy:.1f}%"

        mins, secs = divmod(int(remaining), 60)
        print(f"  Epoch {epoch:>3d}/{epochs}  |  Loss: {avg_loss:.4f}{acc_str}"
              f"  |  {epoch_time:.1f}s/epoch  |  ETA: {mins}m{secs:02d}s")

    return avg_loss


# ── Generation ────────────────────────────────────────────────────

def generate_response(model, tokenizer, context_tokens, device,
                      max_len=50, temperature=0.7, top_k=20):
    model.eval()
    tokens = list(context_tokens)
    end_id = tokenizer.word_to_id['<end>']
    unk_id = tokenizer.word_to_id['<unk>']
    user_id = tokenizer.word_to_id['<user>']
    max_ctx = 128

    with torch.no_grad():
        for _ in range(max_len):
            input_ids = torch.tensor([tokens[-max_ctx:]], dtype=torch.long, device=device)
            logits = model(input_ids)
            next_logits = logits[0, -1, :]
            next_logits = next_logits / temperature

            # Suppress special tokens
            next_logits[0] = float('-inf')        # <pad>
            next_logits[unk_id] = float('-inf')   # <unk>
            next_logits[user_id] = float('-inf')  # <user> (bot shouldn't generate this)

            # Repetition penalty: reduce score of recently generated tokens
            recent = tokens[-15:] if len(tokens) > 15 else tokens
            for t in set(recent):
                next_logits[t] = next_logits[t] * 0.5  # penalize repeats

            if top_k > 0:
                top_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < top_vals[-1]] = float('-inf')

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token == end_id:
                break
            tokens.append(next_token)

    return tokens


# ── Chat Interface ────────────────────────────────────────────────

def chat(model, tokenizer, device):
    print("\n" + "=" * 60)
    print("  CHATBOT v2 — Trained on 50K conversations")
    print("  Type your message. 'quit' to exit, 'reset' to clear.")
    print("=" * 60)

    user_id = tokenizer.word_to_id['<user>']
    bot_id = tokenizer.word_to_id['<bot>']
    history = []

    while True:
        try:
            user_input = input("\n  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == 'quit':
            print("  Goodbye!")
            break
        if user_input.lower() == 'reset':
            history = []
            print("  [History cleared]")
            continue

        user_tokens = tokenizer.encode(user_input)
        context = history + [user_id] + user_tokens + [bot_id]

        output_tokens = generate_response(model, tokenizer, context, device,
                                          max_len=40, temperature=0.7, top_k=20)

        bot_response_tokens = output_tokens[len(context):]
        response_text = tokenizer.decode(bot_response_tokens)
        print(f"  Bot: {response_text}")

        history = output_tokens


# ── Main ──────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = get_device()
    print(f"\n  Device: {device}")

    # Config — bigger model, smaller vocab, more training
    SEQ_LENGTH = 64
    EMBED_DIM = 256      # doubled: richer word representations
    NUM_HEADS = 8        # more attention heads
    NUM_LAYERS = 6       # deeper network
    FF_DIM = 1024        # wider feed-forward layers
    MAX_VOCAB = 5000     # smaller vocab = easier to learn
    EPOCHS = 150         # much more training
    BATCH_SIZE = 32      # smaller batches for better gradients
    LR = 0.0003          # stable learning rate

    # ── Step 1: Tokenize ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 1: TOKENIZATION")
    print("=" * 60)

    # Load QA dataset + hand-crafted conversations
    print("  Loading corpus_qa.txt...")
    with open("corpus_qa.txt", "r", encoding="utf-8") as f:
        qa_corpus = f.read()
    full_corpus = CONVERSATIONS_SMALL + "\n" + qa_corpus
    print(f"  Corpus loaded: {len(full_corpus):,} characters")

    tokenizer = Tokenizer(max_vocab=MAX_VOCAB)
    tokenizer.build_vocab(full_corpus)
    token_ids = tokenizer.encode(full_corpus)

    print(f"\n  Vocabulary size : {tokenizer.vocab_size:,} words (capped at {MAX_VOCAB})")
    print(f"  Total tokens    : {len(token_ids):,}")

    unk_count = sum(1 for t in token_ids if t == tokenizer.word_to_id['<unk>'])
    print(f"  Unknown tokens  : {unk_count:,} ({unk_count/len(token_ids)*100:.1f}%)")

    # ── Step 2: Prepare Data ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 2: CREATING TRAINING DATA")
    print("=" * 60)

    inputs, targets = create_training_data(token_ids, seq_length=SEQ_LENGTH)
    print(f"\n  Sequence length : {SEQ_LENGTH}")
    print(f"  Training pairs  : {len(inputs):,}")

    # ── Step 3: Build Model ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 3: BUILDING TRANSFORMER")
    print("=" * 60)

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        max_len=max(128, SEQ_LENGTH * 2),
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Model will run on: {device}")

    # ── Step 4: Train ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  STEP 4: TRAINING ({EPOCHS} epochs on {device})")
    print("=" * 60)

    start = time.time()
    train_model(model, inputs, targets, device, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE)
    elapsed = time.time() - start
    mins, secs = divmod(int(elapsed), 60)
    print(f"\n  Total training time: {mins}m {secs}s")

    # ── Step 5: Test ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 5: TEST RESPONSES")
    print("=" * 60)

    test_prompts = [
        "hello",
        "what is the sun",
        "tell me about animals",
        "what is ai",
        "i am sad",
        "tell me a fact",
        "what is energy",
        "goodbye",
    ]

    user_id = tokenizer.word_to_id['<user>']
    bot_id = tokenizer.word_to_id['<bot>']

    for prompt in test_prompts:
        context = [user_id] + tokenizer.encode(prompt) + [bot_id]
        output = generate_response(model, tokenizer, context, device,
                                   max_len=30, temperature=0.7, top_k=20)
        response = tokenizer.decode(output[len(context):])
        print(f"\n  You: {prompt}")
        print(f"  Bot: {response}")

    # ── Step 6: Save ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 6: SAVING MODEL")
    print("=" * 60)

    # Move to CPU for saving
    model.to('cpu')
    torch.save({
        'model_state': model.state_dict(),
        'vocab': tokenizer.word_to_id,
        'config': {
            'vocab_size': tokenizer.vocab_size,
            'embed_dim': EMBED_DIM,
            'num_heads': NUM_HEADS,
            'num_layers': NUM_LAYERS,
            'ff_dim': FF_DIM,
            'seq_length': SEQ_LENGTH,
        }
    }, "chatbot_v3_qa.pth")

    print(f"\n  Saved to chatbot_v3_qa.pth")
    print(f"  Parameters: {total_params:,}")
    print(f"  Vocab size: {tokenizer.vocab_size:,}")

    # ── Step 7: Chat ───────────────────────────────────────────────
    model.to(device)
    print("\n" + "=" * 60)
    print("  STEP 7: INTERACTIVE CHAT")
    print("=" * 60)

    chat(model, tokenizer, device)


if __name__ == "__main__":
    main()

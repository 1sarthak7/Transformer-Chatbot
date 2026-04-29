"""
Transformer Chatbot — A GPT-style language model built from scratch.

Evolution of your AI journey:
  Model 1 (first.py):           Numbers -> f(x) = 2x           (2 params)
  Model 2 (text_predictor.py):  Words   -> LSTM next-word       (260K params)
  Model 3 (THIS FILE):          Chat    -> Transformer chatbot  (~1M params)

What changed:
  LSTM (old):        Processes words one-by-one left to right, forgets long context.
  Transformer (new): All words attend to each other simultaneously via attention.

Key concept — Self-Attention:
  "The cat sat on the ___"
  - LSTM processes left-to-right, one step at a time
  - Transformer: every word looks at every other word AT ONCE
    "cat" attends to "sat" (what did the cat do?)
    "on" attends to "sat" and "mat" (where?)

Architecture:
  Tokens -> Embedding + Position -> [Self-Attention -> FFN] x N -> Linear -> Next Word

This is the SAME architecture as GPT-1/2/3/4, just miniature.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from collections import Counter
from corpus import CONVERSATIONS


# PART 1: TOKENIZER (with special conversation tokens)


class Tokenizer:
    """
    Converts words <-> numbers. Now with special conversation tokens.

    Special tokens:
        <pad>  = 0   Padding for shorter sequences
        <unk>  = 1   Unknown words
        <user> = 2   Marks the start of user input
        <bot>  = 3   Marks the start of bot response
        <end>  = 4   Marks end of a conversation turn
    """

    SPECIAL_TOKENS = ['<pad>', '<unk>', '<user>', '<bot>', '<end>']

    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0

    def build_vocab(self, text):
        """Build vocabulary from corpus."""
        words = text.lower().replace('.', '').replace(',', '').replace('?', '').replace('!', '').split()

        # Start with special tokens
        self.word_to_id = {tok: i for i, tok in enumerate(self.SPECIAL_TOKENS)}
        idx = len(self.SPECIAL_TOKENS)

        # Add all unique words
        word_counts = Counter(words)
        for word, _ in word_counts.most_common():
            if word not in self.word_to_id:
                self.word_to_id[word] = idx
                idx += 1

        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.vocab_size = len(self.word_to_id)

    def encode(self, text):
        """Text -> list of token IDs."""
        words = text.lower().replace('.', '').replace(',', '').replace('?', '').replace('!', '').split()
        return [self.word_to_id.get(w, self.word_to_id['<unk>']) for w in words]

    def decode(self, ids):
        """List of token IDs -> text."""
        return ' '.join(self.id_to_word.get(i, '<unk>') for i in ids)

    def decode_one(self, token_id):
        """Single token ID -> word."""
        return self.id_to_word.get(token_id, '<unk>')



# PART 2: TRANSFORMER ARCHITECTURE


class PositionalEncoding(nn.Module):
    """
    Tells the model WHERE each word is in the sequence.

    Transformers process all words in parallel (unlike LSTMs),
    so they need positional info injected explicitly.

    Uses sinusoidal functions:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Each position gets a unique "fingerprint" vector.
    """

    def __init__(self, embed_dim, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention with causal mask.

    THIS IS THE CORE INNOVATION OF TRANSFORMERS.

    How it works (for each word in the sequence):
        1. Create three vectors: Query (Q), Key (K), Value (V)
        2. Q asks: "What am I looking for?"
        3. K says:  "This is what I contain"
        4. V says:  "This is the info to pass along"
        5. Attention = softmax(Q * K^T / sqrt(d)) * V
           (How much should each word attend to every other word?)

    Multi-head = run this attention multiple times in parallel,
    each head can focus on different relationships.

    Causal mask = each word can only attend to PREVIOUS words
    (can't cheat by looking at future words during generation).
    """

    def __init__(self, embed_dim, num_heads, max_len=512, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim

        # Q, K, V projections (all in one matrix for efficiency)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Causal mask — prevents attending to future tokens
        mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer('mask', mask)

    def forward(self, x):
        B, T, C = x.shape  # batch, sequence_length, embed_dim

        # Compute Q, K, V for all heads at once
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores: how much each word attends to every other
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale  # (B, heads, T, T)

        # Apply causal mask (future tokens get -inf -> 0 after softmax)
        attn = attn.masked_fill(self.mask[:T, :T], float('-inf'))

        # Softmax -> attention weights (probabilities)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum of values
        out = attn @ v  # (B, heads, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, C)  # recombine heads

        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """
    One Transformer block = Self-Attention + Feed-Forward + LayerNorm.

    Data flow:
        x -> LayerNorm -> Self-Attention -> + residual -> LayerNorm -> FFN -> + residual -> out
              (Pre-Norm architecture, same as GPT-2)
    """

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),              # Modern activation (smoother than ReLU)
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention with residual connection
        x = x + self.dropout(self.attn(self.ln1(x)))
        # Feed-forward with residual connection
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """
    A miniature GPT model.

    Architecture:
        Token Embedding + Positional Encoding
        -> N x TransformerBlock
        -> LayerNorm
        -> Linear (to vocab size)

    This is structurally identical to GPT-2, just smaller:
        GPT-2:    768 embed, 12 heads, 12 layers,  117M params
        MiniGPT:  128 embed,  4 heads,  4 layers,   ~1M params
    """

    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=4,
                 ff_dim=512, max_len=128, dropout=0.1):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_encoding = PositionalEncoding(embed_dim, max_len)
        self.dropout = nn.Dropout(dropout)

        # Stack of Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.ln_final = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)

        # Weight tying: share embedding and output weights (like GPT-2)
        self.output.weight = self.token_embedding.weight

    def forward(self, x):
        # x: (batch, seq_len) of token IDs
        x = self.token_embedding(x)      # -> (batch, seq_len, embed_dim)
        x = self.position_encoding(x)    # + positional info
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)                 # -> through N transformer layers

        x = self.ln_final(x)             # final layer norm
        logits = self.output(x)          # -> (batch, seq_len, vocab_size)
        return logits



# PART 3: DATA PREPARATION


def create_training_data(token_ids, seq_length=32):
    """
    Create (input, target) pairs for next-token prediction.

    For each position, the target is the NEXT token:
        Input:  [<user>, what, is, the, sun]
        Target: [what,   is,   the, sun, <bot>]

    The model learns to predict every next token simultaneously
    (unlike the LSTM version which only predicted one).
    """
    inputs = []
    targets = []

    for i in range(0, len(token_ids) - seq_length, seq_length // 2):
        chunk = token_ids[i : i + seq_length + 1]
        if len(chunk) < seq_length + 1:
            break
        inputs.append(chunk[:-1])    # all tokens except last
        targets.append(chunk[1:])    # all tokens except first

    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


# PART 4: TRAINING


def train_model(model, inputs, targets, epochs=500, lr=0.001):
    """Train with AdamW optimizer and learning rate warmup."""
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore <pad>
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Learning rate scheduler: warmup then cosine decay
    warmup_epochs = min(50, epochs // 10)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    dataset_size = len(inputs)
    batch_size = min(32, dataset_size)
    print_every = max(1, epochs // 15)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        num_batches = 0

        indices = torch.randperm(dataset_size)
        for start in range(0, dataset_size, batch_size):
            batch_idx = indices[start:start + batch_size]
            batch_in = inputs[batch_idx]
            batch_tgt = targets[batch_idx]

            # Forward: predict next token at every position
            logits = model(batch_in)  # (batch, seq_len, vocab_size)
            loss = criterion(logits.reshape(-1, logits.size(-1)), batch_tgt.reshape(-1))

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / num_batches

        if epoch % print_every == 0 or epoch == 1:
            # Calculate accuracy
            model.eval()
            with torch.no_grad():
                all_logits = model(inputs)
                mask = targets != 0  # ignore padding
                preds = all_logits.argmax(dim=-1)
                correct = ((preds == targets) & mask).sum().item()
                total = mask.sum().item()
                accuracy = correct / total * 100 if total > 0 else 0
            current_lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:>4d}/{epochs}  |  Loss: {avg_loss:.4f}"
                  f"  |  Accuracy: {accuracy:.1f}%  |  LR: {current_lr:.6f}")

    return avg_loss



# PART 5: TEXT GENERATION


def generate_response(model, tokenizer, context_tokens, max_len=50,
                      temperature=0.7, top_k=20):
    """
    Generate text autoregressively with top-k sampling.

    Process:
        1. Feed context tokens to model
        2. Model outputs logits for next token
        3. Apply temperature (controls randomness)
        4. Keep only top-k most likely tokens
        5. Sample from them
        6. Append and repeat until <end> or max_len
    """
    model.eval()
    tokens = list(context_tokens)
    end_id = tokenizer.word_to_id['<end>']
    max_ctx = 128  # model's max sequence length

    with torch.no_grad():
        for _ in range(max_len):
            # Use last max_ctx tokens as input
            input_ids = torch.tensor([tokens[-max_ctx:]], dtype=torch.long)
            logits = model(input_ids)
            next_logits = logits[0, -1, :]  # logits for next token

            # Temperature scaling
            next_logits = next_logits / temperature

            # Top-k filtering: zero out everything except top-k
            if top_k > 0:
                top_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                threshold = top_vals[-1]
                next_logits[next_logits < threshold] = float('-inf')

            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            # Stop at <end>
            if next_token == end_id:
                break

            tokens.append(next_token)

    return tokens



# PART 6: CHATBOT INTERFACE


def chat(model, tokenizer):
    """
    Interactive chatbot with conversation memory.

    How it works:
        1. Maintains full conversation history as token IDs
        2. Each user message is prefixed with <user>
        3. Bot response is generated after <bot> marker
        4. History accumulates (model sees prior context)

    This is exactly how ChatGPT works:
        <user> Hello <bot> Hi there! <user> What is AI? <bot> AI is...
    """
    print("\n" + "=" * 60)
    print("  CHATBOT READY")
    print("  Type your message and press Enter.")
    print("  Type 'quit' to exit, 'reset' to clear history.")
    print("=" * 60)

    user_id = tokenizer.word_to_id['<user>']
    bot_id = tokenizer.word_to_id['<bot>']
    history = []  # conversation memory

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

        # Build context: history + <user> new_message <bot>
        user_tokens = tokenizer.encode(user_input)
        context = history + [user_id] + user_tokens + [bot_id]

        # Generate response
        output_tokens = generate_response(model, tokenizer, context,
                                          max_len=40, temperature=0.7, top_k=15)

        # Extract just the bot's response (tokens after the last <bot>)
        bot_response_tokens = output_tokens[len(context):]
        response_text = tokenizer.decode(bot_response_tokens)

        print(f"  Bot: {response_text}")

        # Update history with this exchange
        history = output_tokens  # keep full context for memory



# MAIN


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    SEQ_LENGTH = 32
    EMBED_DIM = 128
    NUM_HEADS = 4
    NUM_LAYERS = 4
    FF_DIM = 512
    EPOCHS = 500

    # ── Step 1: Tokenize ───────────────────────────────────────────
    print("=" * 60)
    print("  STEP 1: TOKENIZATION")
    print("=" * 60)

    tokenizer = Tokenizer()
    tokenizer.build_vocab(CONVERSATIONS)
    token_ids = tokenizer.encode(CONVERSATIONS)

    print(f"\n  Vocabulary size  : {tokenizer.vocab_size} words")
    print(f"  Total tokens     : {len(token_ids)}")
    print(f"  Special tokens   : {Tokenizer.SPECIAL_TOKENS}")

    # Show examples
    print(f"\n  Sample encodings:")
    for word in ['<user>', '<bot>', '<end>', 'the', 'cat', 'hello', 'ai']:
        wid = tokenizer.word_to_id.get(word, '?')
        print(f"    '{word}' -> {wid}")

    # ── Step 2: Prepare Data ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 2: CREATING TRAINING DATA")
    print("=" * 60)

    inputs, targets = create_training_data(token_ids, seq_length=SEQ_LENGTH)
    print(f"\n  Sequence length  : {SEQ_LENGTH}")
    print(f"  Training pairs   : {len(inputs)}")

    print(f"\n  Example (input -> target at each position):")
    sample_in = tokenizer.decode(inputs[0][:8].tolist())
    sample_tgt = tokenizer.decode(targets[0][:8].tolist())
    print(f"    Input:  {sample_in}")
    print(f"    Target: {sample_tgt}")

    # Step 3: Build Model
    print("\n" + "=" * 60)
    print("  STEP 3: BUILDING TRANSFORMER")
    print("=" * 60)

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        max_len=128,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  {model}")
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  vs LSTM version:  259,795")
    print(f"  vs GPT-2:         117,000,000")

    # ── Step 4: Train ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  STEP 4: TRAINING ({EPOCHS} epochs)")
    print("  LSTM -> Transformer upgrade in action...")
    print("=" * 60, "\n")

    train_model(model, inputs, targets, epochs=EPOCHS, lr=0.001)

    # ── Step 5: Test Generation ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 5: TEST RESPONSES")
    print("=" * 60)

    test_prompts = [
        "hello",
        "what is the sun",
        "tell me a joke",
        "what is ai",
        "i am sad",
        "what is a cat",
        "tell me about space",
        "goodbye",
    ]

    user_id = tokenizer.word_to_id['<user>']
    bot_id = tokenizer.word_to_id['<bot>']

    for prompt in test_prompts:
        context = [user_id] + tokenizer.encode(prompt) + [bot_id]
        output = generate_response(model, tokenizer, context,
                                   max_len=30, temperature=0.7, top_k=15)
        response = tokenizer.decode(output[len(context):])
        print(f"\n  You: {prompt}")
        print(f"  Bot: {response}")

    # ── Step 6: Save ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 6: SAVING MODEL")
    print("=" * 60)

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
    }, "chatbot_model.pth")

    print(f"\n  Saved to chatbot_model.pth")
    print(f"  Parameters: {total_params:,}")

    # ── Step 7: Interactive Chat ───────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 7: INTERACTIVE CHAT")
    print("=" * 60)

    chat(model, tokenizer)


if __name__ == "__main__":
    main()

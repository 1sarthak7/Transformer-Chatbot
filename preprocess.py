"""
Preprocessor — Converts raw train.txt into chatbot training data.

Takes 50K clean sentences from train.txt and converts them into
<user>...<bot>...<end> conversation format for the Transformer.
"""

import random
import re

random.seed(42)

INPUT_FILE = "train.txt"
OUTPUT_FILE = "corpus_large.py"
TARGET_LINES = 50000

# Question templates to wrap statements as conversations
QUESTION_TEMPLATES = [
    "tell me something",
    "tell me a fact",
    "tell me something interesting",
    "share a fact",
    "what do you know",
    "teach me something",
    "give me information",
    "tell me more",
    "what can you tell me",
    "share something interesting",
]

# Topic-specific question patterns (matched by keywords)
TOPIC_QUESTIONS = {
    # Science
    frozenset(["cell", "cells", "biology", "organism", "species"]): [
        "tell me about biology", "what do you know about life science",
    ],
    frozenset(["planet", "star", "solar", "space", "orbit", "galaxy"]): [
        "tell me about space", "what do you know about the universe",
    ],
    frozenset(["water", "ocean", "river", "lake", "sea"]): [
        "tell me about water", "what do you know about the ocean",
    ],
    frozenset(["brain", "neuron", "memory", "cognitive"]): [
        "tell me about the brain", "how does the brain work",
    ],
    frozenset(["energy", "power", "electric", "solar", "wind"]): [
        "tell me about energy", "what is energy",
    ],
    frozenset(["climate", "weather", "temperature", "rain", "snow"]): [
        "tell me about weather", "what do you know about climate",
    ],
    frozenset(["food", "eat", "cook", "meal", "diet", "nutrition"]): [
        "tell me about food", "what should i eat",
    ],
    frozenset(["child", "children", "kid", "kids", "student", "school"]): [
        "tell me about education", "what about children",
    ],
    frozenset(["health", "disease", "medical", "doctor", "patient"]): [
        "tell me about health", "what do you know about medicine",
    ],
    frozenset(["history", "ancient", "century", "war", "empire"]): [
        "tell me about history", "what happened in the past",
    ],
    frozenset(["computer", "software", "technology", "digital", "internet"]): [
        "tell me about technology", "what do you know about computers",
    ],
    frozenset(["animal", "animals", "bird", "fish", "dog", "cat"]): [
        "tell me about animals", "what do you know about animals",
    ],
    frozenset(["plant", "tree", "forest", "flower", "garden"]): [
        "tell me about plants", "what do you know about nature",
    ],
    frozenset(["music", "song", "art", "paint", "creative"]): [
        "tell me about art", "what do you know about music",
    ],
    frozenset(["book", "read", "story", "write", "author"]): [
        "tell me about books", "what should i read",
    ],
}


def is_clean_sentence(line):
    """Filter for clean, usable sentences."""
    line = line.strip()
    if not line:
        return False
    # Length filters
    words = line.split()
    if len(words) < 6 or len(words) > 35:
        return False
    # Must start with letter
    if not line[0].isalpha():
        return False
    # Skip lines with too many special characters
    special = sum(1 for c in line if c in '{}[]<>|\\@#$%^&*~`')
    if special > 2:
        return False
    # Skip lines that are mostly uppercase (headers)
    upper_ratio = sum(1 for c in line if c.isupper()) / len(line)
    if upper_ratio > 0.4:
        return False
    # Skip URLs
    if 'http' in line or 'www.' in line or '.com' in line:
        return False
    # Skip lines with numbers that look like citations/references
    if re.search(r'\[\d+\]', line) or re.search(r'\(\d{4}\)', line):
        return False
    return True


def clean_text(text):
    """Clean a sentence for training."""
    text = text.strip().lower()
    # Remove common punctuation but keep basic structure
    text = re.sub(r'["""''`]', '', text)
    text = re.sub(r'[;:!?]', '', text)
    text = re.sub(r'\(.*?\)', '', text)  # remove parenthetical
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove trailing period
    text = text.rstrip('.')
    return text


def pick_question(sentence):
    """Pick a relevant question template based on sentence content."""
    words = set(sentence.lower().split())
    for keywords, questions in TOPIC_QUESTIONS.items():
        if words & keywords:
            return random.choice(questions)
    return random.choice(QUESTION_TEMPLATES)


def main():
    print(f"Reading {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        all_lines = f.readlines()

    print(f"Total lines in file: {len(all_lines):,}")

    # Filter clean sentences
    print("Filtering clean sentences...")
    clean_lines = [line for line in all_lines if is_clean_sentence(line)]
    print(f"Clean sentences found: {len(clean_lines):,}")

    # Sample target amount
    if len(clean_lines) > TARGET_LINES:
        sampled = random.sample(clean_lines, TARGET_LINES)
    else:
        sampled = clean_lines
    print(f"Sampled: {len(sampled):,} sentences")

    # Convert to conversation format
    print("Converting to conversation format...")
    conversations = []
    for line in sampled:
        cleaned = clean_text(line)
        if len(cleaned.split()) < 5:
            continue
        question = pick_question(cleaned)
        conv = f"<user> {question} <bot> {cleaned} <end>"
        conversations.append(conv)

    # Also keep original hand-crafted conversations
    print(f"Final conversation count: {len(conversations):,}")

    # Write output
    print(f"Writing {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('"""\nLarge conversational dataset — auto-generated from train.txt\n')
        f.write(f'Contains {len(conversations):,} conversation pairs.\n"""\n\n')
        f.write('CONVERSATIONS_LARGE = """\n')
        for conv in conversations:
            f.write(conv + '\n')
        f.write('"""\n')

    # Stats
    all_text = '\n'.join(conversations)
    total_words = len(all_text.split())
    unique_words = len(set(all_text.lower().split()))
    print(f"\nDataset Stats:")
    print(f"  Conversations : {len(conversations):,}")
    print(f"  Total words   : {total_words:,}")
    print(f"  Unique words  : {unique_words:,}")
    print(f"  Avg words/conv: {total_words // len(conversations)}")
    print(f"\nDone! Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

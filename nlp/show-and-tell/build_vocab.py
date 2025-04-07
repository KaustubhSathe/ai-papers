import nltk
import pickle
import argparse
from collections import Counter
import json
import os

# Ensure NLTK 'punkt' tokenizer and related resources are available
try:
    nltk.data.find('tokenizers/punkt')
    # Check for punkt_tab as well, since word_tokenize might need it internally
    nltk.data.find('tokenizers/punkt_tab')
except LookupError: # nltk.data.find raises LookupError if resource not found
    print("NLTK 'punkt' or 'punkt_tab' resource not found. Downloading...")
    nltk.download('punkt')
    # Also download 'punkt_tab' as it seems required by the tokenizer process
    nltk.download('punkt_tab')

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        # Add special tokens upon initialization
        self.add_word('<pad>') # Padding token
        self.add_word('<start>') # Start-of-sequence token
        self.add_word('<end>')   # End-of-sequence token
        self.add_word('<unk>')   # Unknown word token

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(json_path, threshold):
    """Build a simple vocabulary wrapper."""
    try:
        with open(json_path, 'r') as f:
            coco = json.load(f)
    except FileNotFoundError:
        print(f"Error: Annotation file not found at {json_path}")
        raise
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}")
        raise

    counter = Counter()
    caption_count = 0
    if 'annotations' not in coco:
        print(f"Error: 'annotations' key not found in {json_path}. Is this a valid COCO captions file?")
        return None # Or raise an error

    for i, ann in enumerate(coco['annotations']):
        caption = ann['caption']
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)
        caption_count += 1

        if (i + 1) % 10000 == 0:
            print(f"[{i+1}/{len(coco['annotations'])}] Tokenized captions.")

    print(f"Tokenized {caption_count} captions.")

    # Filter words based on the frequency threshold
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    print(f"Total unique words found: {len(counter)}")
    print(f"Vocabulary size (frequency >= {threshold}): {len(words)}")

    # Create a vocab wrapper and add the words
    vocab = Vocabulary()
    # Special tokens are added in __init__

    for word in words:
        vocab.add_word(word)

    # Ensure special tokens are present (they should be, due to __init__)
    assert '<pad>' in vocab.word2idx
    assert '<start>' in vocab.word2idx
    assert '<end>' in vocab.word2idx
    assert '<unk>' in vocab.word2idx

    return vocab

def main(args):
    # Ensure the directory for the vocab path exists
    vocab_dir = os.path.dirname(args.vocab_path)
    if vocab_dir and not os.path.exists(vocab_dir):
        os.makedirs(vocab_dir)
        print(f"Created directory: {vocab_dir}")

    print(f"Building vocabulary from {args.caption_path} with threshold {args.threshold}...")
    vocab = build_vocab(json_path=args.caption_path, threshold=args.threshold)

    if vocab:
        with open(args.vocab_path, 'wb') as f:
            pickle.dump(vocab, f)
        print(f"Vocabulary saved to {args.vocab_path}")
        print(f"Total vocabulary size: {len(vocab)}")
    else:
        print("Vocabulary building failed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Default paths assume a 'data/coco/annotations' structure, adjust as needed
    parser.add_argument('--caption_path', type=str,
                        default='./annotations/captions_train2017.json',
                        help='Path for train annotation file.')
    parser.add_argument('--vocab_path', type=str, default='./vocab.pkl',
                        help='Path for saving vocabulary wrapper.')
    parser.add_argument('--threshold', type=int, default=4,
                        help='Minimum word count threshold.')
    args = parser.parse_args()
    main(args)

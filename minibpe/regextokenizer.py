import regex as re # regex is a more powerful module than the built-in re
import tiktoken # official library for tokenization from openAI
from basictokenizer import BasicTokenizer, get_stats, merge

# GPT-2 (does not merge spaces)
enc  = tiktoken.get_encoding("gpt2")
print(enc.encode("    hello world!!!"))

# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# Forced splits using regex patterns (GPT series)
gpt2pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+""")
gpt4pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+""")

print(re.findall(gpt2pat, "Hello've world123 how's are you"))
print(re.findall(gpt4pat, "Hello've world123 how's are you"))

"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Unlike BasicTokenizer:
- RegexTokenizer handles an optional regex splitting pattern.
- RegexTokenizer handles optional special tokens.
"""

class RegexTokenizer(BasicTokenizer):
    def __init__(self, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size, verbose=False):
        # Tokenizer can train a vocabulary of size vocab_size from text
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)
        # input text preprocessing
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        if verbose:
            # print(text)
            print("length:", len(text))
            print(ids)
            print("length:", len(ids))

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes

        for i in range(num_merges):
            # count the number of times every consecutive pair appears
            stats = {}
            for chunk_ids in ids:
                # passing in stats will update it in place, adding up counts
                get_stats(chunk_ids, stats)
            
            if stats: # guard
                # find the pair with the highest count
                pair = max(stats, key=stats.get)
                # mint a new token: assign it the next available id
                idx = 256 + i
                # replace all occurrences of pair in ids with idx
                ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
                # save the merge
                merges[pair] = idx
                vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
                # prints
                if verbose:
                    print(f"Merging {pair} into a new token {idx}")

        if verbose:
            print("Training complete.")
            print("Final vocab size:", len(vocab))
            print(f"Compression ratio: {len(text_chunks) / len(ids):.2f}X")

        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

    def register_special_tokens(self, special_tokens: dict[str]) -> str:
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def _encode_chunk(self, text_bytes):
        # return the token ids
        # Tokenizer can encode a string into a list of integers
        ids = list(text_bytes) # list of integers in range 0..255
        while len(ids) >= 2: # more consise than while True, explicitly checking if at least 2 tokens
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids
    
    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids
    
    def encode(self, text, allowed_special="none_raise"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """

        # decode the user desire w.r.t handling of special tokens
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            # shortcut: if no special tokens, just use ordinary encoding
            return self.encode_ordinary(text)
        # otherwise, we have to be careful with potential special tokens in text
        # we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens
        # we can use re.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        # now all the special characters are separated from the rest of the text
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                # this is an odinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part))
        return ids
  
    def decode(self, ids):
        # Tokenizer can decode a list of integers into a string
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

vocab_size = 30000
# GPT-2 (does not merge spaces)
enc2  = tiktoken.get_encoding("gpt2")
print(enc2.encode("    hello world!!!"))

# GPT-4 (merge spaces)
enc4 = tiktoken.get_encoding("cl100k_base") # this is the GPT-4 tokenizer
ids = enc4.encode("hello world!!!? (안녕하세요!) lol123 😉")
devtext = enc4.decode(ids) # get the same text back

# initialize regex_tokenizer
regex_tokenizer = RegexTokenizer()

# train on 
text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."
# read to inspect
with open('tests/taylorswift.txt', 'r', encoding='utf-8') as f:
    taylor = f.read()

regex_tokenizer.train(text+taylor, vocab_size)
regex_tokenizer.save('regex')
regex_tokenizer.load('regex.model')

regexdevtext = regex_tokenizer.decode(regex_tokenizer.encode("hello world!!!? (안녕하세요!) lol123 😉"))
print(regexdevtext == devtext)
# bigger test using text 
# Encode with GPT-4 tokenizer

# test on text
gpttext = enc4.decode(enc4.encode(text))
regextext = regex_tokenizer.decode(regex_tokenizer.encode(text))
print(regextext == gpttext)

# test on taylorswift.txt wikipedia page on taylor swift
gpttext = enc4.decode(enc4.encode(taylor))
regextext = regex_tokenizer.decode(regex_tokenizer.encode(taylor))
print(regextext == gpttext)
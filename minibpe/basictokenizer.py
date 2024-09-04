import unicodedata

# python strings are immutable sequences of unicode code points
# unicode is a definition for text-encoding as numbers
# python builtin function ord()
print([ord("A")])
print([ord(x) for x in "Hello World!"])
# why not use ord(). one reason is vocabulary will be very long.
# more worrying reason, unicode is always changing updating, not very stable

# unicode defines 3 types of encodings, utf-8 (most popular), utf-16, utf-32
# utf-8 --> 1 to 4 bytes
print(list("Hello World!".encode("utf-8")))
print(list("Hello World!".encode("utf-16"))) 
print(list("Hello World!".encode("utf-32")))
# wastefullness of utf-16 and utf-32, many zeros
# but we cannot use it naively,
# utf-8 are byte-strings, implying a vocabulary length of 256
# all of our text will be stretched out over very long sequences of bytes
# embedding table tiny, prediction final layer tiny
# finite context length in a transformer
# attention is expensive
# won't work for very long text for the next token prediction task
# tokenization free would be amazing, feed raw bytes for our models

# compress the bytes using the Byte Pair Encoding (BPE) algorithm
# iteratively find the pair of byte that occur the most frequently
# replace that pair with a new byte we append to the data
#XdXac
#X=ZY
#Y=ab
#Z=aa
# iteratively find the pair of tokens that occur the most frequently
# replace that pair with a new token we append to the vocabulary

# most common pair of same two consecutive number
def mcp2same_consecutive(nums):
    pairs = {}
    for i in range(len(nums) - 1):
        if nums[i] == nums[i+1]:
            pair = (nums[i], nums[i+1])
            pairs[pair] = pairs.get(pair, 0) + 1
    if not pairs:
        return None  # Return None if no pairs are found
    return max(pairs, key=pairs.get)

# function that finds the most common pair
# for i in range(len(bytes) - 1):
#     pair = (bytes[i], bytes[i + 1])
#     counts[pair] = counts.get(pair, 0) + 1

# tokens = text.encode("utf-8")
# tokens = list(map(int, tokens))
# print(counts)
# print(sorted(((v, k) for k,v in stats.items()), reverse=True))
# print(chr(101), chr(32))
# print(mcp2same_consecutive(tokens))
# max(dict, func)
# top_pair = max(stats, key=stats.get)

# minbpe
# Byte Pair Encoding (BPE) algorithm tokenization
"""
Contains the base Tokenizer class and a few common helper functions.
The base class also contains the (common) save/load functionality.
It would be possible to be a lot more strict about the interface and
e.g. isolating all regex/pattern parts to the RegexTokenizer, but
some concessions are made for simplicity.
"""
# -----------------------------------------------------------------------------
# a few helper functions useful for both BasicTokenizer and RegexTokenizer

def get_stats(ids, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    # Calculate the frequency of pairs of tokens.
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    # in the list of ints (ids), replace all consecutive occurences of pair with the new token idx
    newids = []
    i = 0
    while i < len(ids):
        # if not at the very last position AND the pair matches, replace it
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

# first two helper functions...
def replace_control_characters(s: str) -> str:
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

class BasicTokenizer:
    def __init__(self):
        # default: vocab size of 256 (all bytes), no merges, no patterns
        self.merges = {} # (int, int) -> int
        self.pattern = "" # str
        self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab = {}

    def train(self, text, vocab_size, verbose=False):
        # Tokenizer can train a vocabulary of size vocab_size from text
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255
        # copy so we don't destroy the original list

        if verbose:
            print(text)
            print("length:", len(text))
            print(ids)
            print("length:", len(ids))

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes

        for i in range(num_merges):
            # count up the number of times every consecutive pair appers
            stats = get_stats(ids)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurences of pair in ids with idx
            ids = merge(ids, pair, idx)
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[1]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"Merging {pair} into a new token {idx}")

        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

        if verbose:
            print("Training complete.")
            print("Final vocab size:", len(self.vocab))
            print(f"Compression ratio: {len(text_bytes) / len(ids):.2f}X")

    def encode(self, text):
        # Tokenizer can encode a string into a list of integers
        text_bytes = text.encode("utf-8") # raw bytes
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
    
    def decode(self, ids):
        # Tokenizer can decode a list of integers into a string
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text
    
    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        # replace bytes([idx]) at idx with byte pair at idx
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab
    
    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        # write the model: to be used in load() later
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # write the version, pattern and merges, that's all that's needed
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            # write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # the merges dict
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        # write the vocab: for human to look at
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx:pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors ='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s} {idx}\n]")
    
    def load(self, model_file):
        """Inverse of save() but only for the model file"""
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "minbpe v1"
            # read the pattern
            self.pattern = f.readline().strip()
            # read the special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()

# test basic tokenizer is working
text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."
v_size = 276 # the desired final vocabulary size

basic_tokenizer = BasicTokenizer()
basic_tokenizer.train(text, vocab_size=v_size, verbose=True)

test_text = "🅤🅝🅘🅒🅞🅓🅔! 😄"
encoded_test = basic_tokenizer.encode(test_text)
decoded_test = basic_tokenizer.decode(encoded_test)
print(decoded_test == test_text)

test_text2 = "Hello World 123"
encoded_test2 = basic_tokenizer.encode(test_text2)
decoded_test2 = basic_tokenizer.decode(encoded_test2)
print(decoded_test2)
print(test_text2)
print(decoded_test2 == test_text2)

basictext = basic_tokenizer.decode(basic_tokenizer.encode(text))
print(basictext == text)
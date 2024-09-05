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

text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."
vocab_size = 276 # the desired final vocabulary size

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
def commonbpe(ids, counts=None):

    counts = {} if counts is None else counts
    for i in range(len(ids) - 1):
        pair = (ids[i], ids[i + 1])
        counts[pair] = counts.get(pair, 0) + 1

text_bytes = text.encode("utf-8")
tokens = list(map(int, text_bytes)) # list of integers in range 0..255
print(counts)
print(sorted(((v, k) for k,v in stats.items()), reverse=True))
print(chr(101), chr(32))
print(mcp2same_consecutive(tokens))
max(dict, func)
top_pair = max(stats, key=stats.get)
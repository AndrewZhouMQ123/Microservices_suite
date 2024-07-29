# HMAC-SHA256 function
import secrets
import hashlib

# K
# cryptographicall secure key
# token_bytes is wrapper function for os.urandom
key0 = secrets.token_bytes(24)

# xor
def xor(x, y):
    return bytes(a ^ b for (a, b) in zip(x, y))

# HMAC implementation using SHA-256 hash function based on rfc2104 standards
# H(K XOR opad, H(K XOR ipad, text))
# (1) append zeros to the end of K to create a B byte string
#         (e.g., if K is of length 20 bytes and B=64, then K will be
#          appended with 44 zero bytes 0x00)
# (2) XOR (bitwise exclusive-OR) the B byte string computed in step
#         (1) with ipad
# (3) append the stream of data 'text' to the B byte string resulting
#         from step (2)
# (4) apply H to the stream generated in step (3)
# (5) XOR (bitwise exclusive-OR) the B byte string computed in
#         step (1) with opad
# (6) append the H result from step (4) to the B byte string
#         resulting from step (5)
# (7) apply H to the stream generated in step (6) and output
#         the result
# (8) truncate the MAC output by slicing

# Recommended that the output length t be not less than half the length of the hash
# output (to match the birthday attack bound) and not less than 80 bits
# (a suitable lower bound on the number of bits that need to be
# predicted by an attacker).  
# Denoting a realization of HMAC that uses a hash function H with t bits of output as HMAC-H-t. 
# For example, HMAC-SHA1-80 denotes HMAC computed using the SHA-1 function
# and with the output truncated to 80 bits. (If the parameter t is not
# specified, e.g. HMAC-MD5, then it is assumed that all the bits of the
# hash are output.)

def hmac_sha256_160(key_K, data):
    block_size = 64
    if len(key_K) > block_size:
        key_K = hashlib.sha256(key_K).digest() # hash the key if too big
    if len(key_K) < 64:
        key_K = key_K + b'\x00' * (block_size - len(key_K)) # add zeros to the key if too small
    ipad = b'\x36' * block_size # ipad = the byte 0x36 repeated B times
    opad = b'\x5c' * block_size # opad = the byte 0x5C repeated B times.
    h_inner = hashlib.sha256(xor(key_K, ipad)) # initialize h_inner value
    h_inner.update(data) # achieves same result as h_inner + data
    h_outer = hashlib.sha256(xor(key_K, opad)) # initialize h_outer value
    h_outer.update(h_inner.digest())
    return h_outer.digest()[:20]

# test vectors for HMAC-SHA-256 based on rfc4321 standards

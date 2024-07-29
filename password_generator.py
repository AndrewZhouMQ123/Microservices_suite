# password generator
import random
import string

# every 30 seconds
# generate password of given length

def generate_password(length=20):
    characters = string.ascii_letters + string.digits + string.punctuation
    return ''.join(random.choice(characters) for _ in range(length))
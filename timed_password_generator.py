# timed password generator
import random
import string
import time
from countdown_timer import countdown

# every 60 seconds
# generate password length of 20

def generate_password(length=20):
    characters = string.ascii_letters + string.digits + string.punctuation
    return ''.join(random.choice(characters) for _ in range(length))

while True:
    password = generate_password()
    print(f"Generated Password: {password}")
    time.sleep(60)
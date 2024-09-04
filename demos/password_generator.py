# password generator
import random
import string

# jlg]Z^m;Sqp1g`c0~_*6UuHpB%BjeYh1
# generate password of given length

def generate_password(length=20):
    characters = string.ascii_letters + string.digits + string.punctuation
    return ''.join(random.choice(characters) for _ in range(length))

# later use int from GUI variable slider 
length = input('Input your password length: ')
length = int(length)
print(generate_password(length))
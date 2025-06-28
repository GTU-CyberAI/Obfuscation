from encrypt import encryption
from decrypt import decryption

if __name__ == "__main__":
    with open("./tests/sample1_deob.txt", "r") as file:
        contents = file.read()
        enc_out = encryption(contents)
        dec_out = decryption(enc_out, contents)

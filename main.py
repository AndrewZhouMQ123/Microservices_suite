# main.py

import customtkinter
from customtkinter import CTkImage
import tkinter as tk
from tkinter import messagebox
from password_generator import generate_password
from PIL import Image

def passkeyUI():
    try:
        password = generate_password()
        label = customtkinter.CTkLabel(master=frame, text=password, font=("Roboto", 20), fg_color="transparent")
        label.pack(padx=10, pady=10)
        print("Password generated successfully")
    except Exception as e:
        print(f"Failed to generate password: {e}")
        messagebox.showerror("Error", f"Failed to generate password: {e}")

# System Settings
customtkinter.set_appearance_mode("system")
customtkinter.set_default_color_theme("dark-blue")

print("Creating main app window...")
# Our app frame
app = customtkinter.CTk()
app.geometry("720x480")
app.title("SimplyAuthenticator")

frame = customtkinter.CTkFrame(master=app)
frame.pack(padx=60, pady=20, fill="both", expand=True)

print("Adding title label...")
# Add UI element
title = customtkinter.CTkLabel(master=frame, text="Add a new authenticator passkey", font=("Roboto", 20))
title.pack(padx=10, pady=10)

print("Loading image...")
# Load and display image
try:
    im = Image.open("behelit.png")
    photo = CTkImage(im)
    print("Image loaded successfully")
except Exception as e:
    print(f"Failed to load image: {e}")
    messagebox.showerror("Error", f"Failed to load image: {e}")
    photo = None

# Generate Password Button
button = customtkinter.CTkButton(master=frame, image=photo, text="Add Passkey", command=passkeyUI, font=("Roboto", 20))
print("Button with image created")
button.pack(padx=10, pady=10)

print("Running app...")
# Run app
app.mainloop()
print("App has exited")
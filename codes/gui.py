# Import the required libraries
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import os

# Create an instance of tkinter frame or window
win = Tk()

# Set the size of the window
win.geometry("700x350")

def select_folder():
   source_path = filedialog.askdirectory(title='Select the Parent Directory')
   
   

button1 = ttk.Button(win, text="Select a Folder", command=select_folder)

button1.pack(pady=5)

win.mainloop()
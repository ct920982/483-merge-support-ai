import tkinter as tk
from tkinter import filedialog

import model

# TODO: dynamic output file name

# for now just have user select input file, makes things faster
def select_file():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select Input File",
        initialdir="../data/raw",
        filetypes=[("Mp4 files", "*.mp4")]
    )
    root.destroy()
    
    return file_path

selected_input = select_file()

if selected_input:
    print(f"File selected: {selected_input}")
    model.run_model(selected_input, "../data/processed/gui_out.mp4")
else:
    print("User cancelled the selection.")
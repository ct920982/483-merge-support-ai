import tkinter as tk
from tkinter import filedialog
from pathlib import Path

from model import UnprotectedLeftTurnFeatureExtraction # import class then initialize here

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
    file_name_extension = Path(selected_input).name

    model = UnprotectedLeftTurnFeatureExtraction()
    model.run_model(selected_input, f"../data/processed/{file_name_extension}")
else:
    print("User cancelled the selection.")
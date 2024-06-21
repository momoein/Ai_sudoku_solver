import numpy as np
import gradio as gr

from model.recognize import recognize_sudoku
from csp.sudoku import solver



def sudoku(input_img):
    map = recognize_sudoku(input_img)
    res = solver(map)
    return "\n".join(["input:", np.array2string(map), "\nresult:", np.array2string(res)])



demo = gr.Interface(sudoku, gr.Image(), "text")

if __name__ == "__main__":
    demo.launch()

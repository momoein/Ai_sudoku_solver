import numpy as np
import gradio as gr

from model.recognize import recognize_sudoku
from csp.sudoku import solver



def sudoku(input_img):
    map = recognize_sudoku(input_img)
    res = solver(map)
    res = np.array2string(res) if res is not None else "not found any result for this sudoku"
    return np.array2string(map), res


def changs(im):
    return im["composite"]



with gr.Blocks() as demo:
    with gr.Row():
        img = gr.ImageEditor(
            scale=2,
            type="numpy",
            crop_size="1:1",
        )
        img_preview = gr.Image(visible=False)
        with gr.Column():
            txt1 = gr.Textbox(lines=10, label="recognizing sudoku", show_label=True)
            txt2 = gr.Textbox(lines=10, label="solving sudoku", show_label=True)
            btn = gr.Button("run")
    img.change(changs, inputs=img, outputs=img_preview, show_progress="hidden")
    btn.click(sudoku, inputs=img_preview, outputs=[txt1, txt2])
    


if __name__ == "__main__":
    demo.launch()

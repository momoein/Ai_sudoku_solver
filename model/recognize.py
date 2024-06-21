import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from model.train import DigitModel



# ---------- Load the pre-trained model ----------
model_path = "./model/digit_model.pth"
model = DigitModel()
model.load_state_dict(torch.load(model_path))
model.eval()


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return threshold

def extract_digit(cell):
    threshold = preprocess_image(cell)
    roi = cv2.resize(threshold, (28, 28), interpolation=cv2.INTER_AREA)
    roi = roi.astype('float32') / 255.0
    roi = transform(roi)
    roi = roi.unsqueeze(0)
    return roi

def recognize_digit(model, cell):
    digit_image = extract_digit(cell)
    if digit_image is None:
        return 0
    with torch.no_grad():
        outputs = model(digit_image)
    _, predicted = torch.max(outputs.data, 1)
    return predicted.item()


# ---------- CV preprocessing ----------

def get_cells(image, lines):
    X, Y = set(), set()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        X.add(x1), X.add(x2)
        Y.add(y1), Y.add(y2)

    X, Y = sorted(list(X)), sorted(list(Y))
    X = [0] + [X[i] for i in range(1, len(X)) if abs(X[i-1] - X[i]) > 5]
    Y = [0] + [Y[i] for i in range(1, len(Y)) if abs(Y[i-1] - Y[i]) > 5]

    cells = []
    for i in range(1, len(X)):
        for j in range(1, len(Y)):
            cells.append(image[X[i-1]:X[i], Y[j-1]:Y[j]])
    
    return cells

def white_frame_edges(image, padding: int):
    assert padding >= 0
    image = image.copy()
    for p in range(padding):
        image[:, p] = np.array(list(map(lambda x: np.array(3*[255]), image[:, p])))
        image[:, -p] = np.array(list(map(lambda x: np.array(3*[255]), image[:, -p])))
        image[p, :] = np.array(list(map(lambda x: np.array(3*[255]), image[p, :])))
        image[-p, :] = np.array(list(map(lambda x: np.array(3*[255]), image[-p, :])))
    return image

def delete_edge_lines(images: list, padding: int=2):
    for i in range(len(images)):
        images[i] = white_frame_edges(images[i], padding)
    return images


def empty_cell(cell):
    return np.mean(cell) > 250


# load image and detect lines
def detect_lines(image):
    # img = cv2.imread(image, cv2.IMREAD_COLOR)
    img = image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=25, maxLineGap=50)
    return img, lines

# extract cells
def extract_cells(image, padding=4):
    """padding : useed to delete padding lines in each cell"""
    img, lines = detect_lines(image)
    cells = get_cells(img, lines)
    cells = delete_edge_lines(cells, padding)
    return cells


# main function
def recognize_sudoku(image:str | np.ndarray):
    """(image) could be image or path to specific image"""
    if isinstance(image, str):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    cells = extract_cells(image)
    sudoku = np.zeros(shape=(9, 9), dtype=str)
    for i in range(len(cells)):
        if i//9 < 9:
            if empty_cell(cells[i]):
                sudoku[i//9, i%9] = "_"
                continue
            predict = recognize_digit(model, cells[i])
            sudoku[i//9, i%9] = predict
    return sudoku


if __name__ == "__main__":
    image_path = "img/sudoku7.png"
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    sudoku = recognize_sudoku(img)
    print(sudoku)
    np.sum(sudoku != "_")
# Ai_sudoku_solver
Ai Sudoku Solver is a simple program to solve sudoku using computer vision and neural networks.
you can simply give an image of sudoku to it and it return an array of completed sudoku.

We have three important components:
1. Computer vision to extract empty cells and numbers in the given image 
2. CNN to recognize each number in the image
3. csp (constraint satisfaction problem) solver to complete sudoku

`NOTE:`
- This project is developed for fun and learning only, it may have bugs.
- The computer vision component is not the best implementation, I'll try and hopefully fix this.
- For now, `keep in mind` that the input image only contains the sudoku table
And I recommend that the rows and columns of the table are not distorted.

## Version : v0.1 Beta
in this model i used MNIST and [printed digit dataset](https://github.com/kaydee0502/printed-digits-dataset).

<img src="https://github.com/momoein/Ai_sudoku_solver/blob/master/data/printed_digit/1/1_0.jpg" width="100" height="100" align = "left"/>
<img src="https://github.com/momoein/Ai_sudoku_solver/blob/master/data/printed_digit/2/2_0.jpg" width="100" height="100" align="middle"/>
<img src="https://github.com/momoein/Ai_sudoku_solver/blob/master/data/printed_digit/8/8_0.jpg" width="100" height="100" align="left"/>
<img src="https://github.com/momoein/Ai_sudoku_solver/blob/master/data/printed_digit/9/9_0.jpg" width="100" height="100" align="middle"/>

## Repository Structure:

+ All images are in [assets](https://github.com/momoein/Ai_sudoku_solver/tree/master/data) 

+ Folder [model](https://github.com/momoein/Ai_sudoku_solver/tree/master/model) contains a pretrained CNN model trained on MNIST (on 20 epochs with batch size 10) and [printed digit dataset](https://github.com/kaydee0502/printed-digits-dataset) (on 10 epochs with batch size 10).

+ Folder [csp](https://github.com/momoein/Ai_sudoku_solver/tree/master/csp) is a helper to convert sudoku map (np.array with shape 9*9) to constraints satisfied problem and solve this.

+ Folder [img](https://github.com/momoein/Ai_sudoku_solver/tree/master/img) contains some input image to test the model.

  <img src="https://github.com/momoein/Ai_sudoku_solver/blob/master/img/sudoku1.jpg" width="225" height="225" align = "left"/>
  <img src="https://github.com/momoein/Ai_sudoku_solver/blob/master/img/sudoku2.jpg" width="225" height="225" align = "middle"/>
  
+ File [main.py](https://github.com/momoein/Ai_sudoku_solver/tree/master/main.py)


## installation

after download or clone repository go to project directory:
```sh
git clone https://github.com/momoein/Ai_sudoku_solver.git
cd Ai_sudoku_solver
```
```sh
pip install -r requirements. txt
``` 
## usage
windows:
```sh
py main.py
```
linux:
```sh
python3 main.py
```
after run open the browser and search this URL: 
```
http://127.0.0.1:7860
```
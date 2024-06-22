# Ai_sudoku_solver

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

  <img src="https://github.com/momoein/Ai_sudoku_solver/blob/master/img/sudoku1.png" width="100" height="225" align = "left"/>
  <img src="https://github.com/momoein/Ai_sudoku_solver/blob/master/img/sudoku2.png" width="100" height="225" align = "middle"/>
  
+ File [main.py](https://github.com/momoein/Ai_sudoku_solver/tree/master/main.py)

## Contribute:
We still required more data, apart from that images of digit `0` is needed, Pull Requests are most welcome and appreciated..



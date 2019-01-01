# Image-scaling
Naive scaling methods are content-unaware methods and hence produce undesirable results. Here is implementation of content-aware scaling of images by identifying and manipulating the low energy seams which most of the times will produce desirable results.
The paper to refer to http://www.eng.tau.ac.il/~avidan/papers/imretFinal.pdf.
This method was proposed by Shai Avidan and Ariel Shamir.
How to use this?
1. Clone the repository.
2. Run the file scale.py on terminal. Copy the following command: python3 scale.py
3. First you'll be asked to input the image filepath. Please type the relative path of that file observed from the cloning directory of the repository.
4. The following input will be interactive, understandable to an average human.
5. The scaled image will be saved in the cloning directory with the file name scaled_<the original filename>.

NOTES:
1. The output is currently grayscaled.
2. The image can be scaled once either heightwise or widthwise per run of the file.

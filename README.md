# ML2

# Lab No. 1
# Machine Learning 2
# Por Juan Esteban Rojas


----------------------------------------------------------------------
----------------------------------------------------------------------
1. Simulate any random rectangular matrix A 
- What is the rank and trace of A?
    - R://
- What is the determinant of A?
    - R: // it is not calculatethe A matrix determinant, beacuse it is a rectangular matrix
- Can you invert A? How?
    - R: // it is not able to invert the A matrix, beacuse it is a rectangular matrix
- How are eigenvalues and eigenvectors of A’A and AA’ related? What interesting differences can you notice between both?
    R: - The eigenvalues are the same for A'A and AA'.A
       - Both are related in the way we can reconstruct the original A matrix through the A'A, AA' eigenvectors and the eigenvalues.
----------------------------------------------------------------------
- // To see task 1 development detais look and run the task1.py script


----------------------------------------------------------------------
----------------------------------------------------------------------
2. Add a steady, well-centered picture of your face to a shared folder alongside your classmates
- Edit your picture to be 256x256 pixels, grayscale (single channel)
- Plot your edited face
- Calculate and plot the average face of the cohort.
- How distant is your face from the average? How would you measure it?
    R:// To know how distant is my face from the average, we will use an euclidean distance, so
          the first step is to substract my face image from the average image, and then calculate 
          the euclidean distance. The calculated distance is: 
----------------------------------------------------------------------
  
- // To see task 2 development details look and run the task2.py script


----------------------------------------------------------------------
----------------------------------------------------------------------
3. Let’s create the unsupervised Python package
- Same API as scikit-learn: fit(), fit_transform(), transform(), hyperparams at init
- Manage dependencies with Pipenv or Poetry
- Implement SVD from scratch using Python and NumPy
- Implement PCA from scratch using Python and NumPy
[https://github.com/rushter/MLAlgorithms/blob/master/mla/pca.py,
https://github.com/patchy631/machine-learning/blob/main/ml_from_scratch/PCA_from_scratch.ip
ynb]
- Implement t-SNE from scratch using Python and NumPy
[https://nlml.github.io/in-raw-numpy/in-raw-numpy-t-sne/]
----------------------------------------------------------------------

- R:// In the project there is the package unsupervised with the modules:
    - svd: allocated on ../unsupervised/svd
    - pca: allocated on ../unsupervised/pca
    - t_sne: allocated on ../unsupervised/t_sne

    - Note: In the task3.py there are some test of each module.


----------------------------------------------------------------------
----------------------------------------------------------------------
4. Apply SVD over the picture of your face, progressively increasing the number of singular values used. 
- Is there any point where you can say the image is appropriately reproduced?
    - R:// Based on the visual inspection of the reconstructed images with different sibgular values used,
          I can say tha the images is appropriately reproduced from 25 or more singular values
- How would you quantify how different your photo and the approximation are?
    - R:// We can determine ghow different my phot is against the aproximated reconstructed image calculating
            the euclidean distance between both images
----------------------------------------------------------------------
- // To see task 4 development details look and run the task4.py script

  
----------------------------------------------------------------------
----------------------------------------------------------------------
5. Train a naive logistic regression on raw MNIST images to distinguish between 0s and 8s. We are calling 
this our baseline. What can you tell about the baseline performance?

- R:// After apply the logistic regression, the accuracy of the trained model is xxxx
----------------------------------------------------------------------
- // To see task 5 development details look and run the task5.py script

----------------------------------------------------------------------
----------------------------------------------------------------------








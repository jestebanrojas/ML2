# ML2

# Lab No. 1
# Machine Learning 2
# Por Juan Esteban Rojas


----------------------------------------------------------------------
----------------------------------------------------------------------
1. Simulate any random rectangular matrix A

With a random created matrix of:
[[2 0 6 3]
 [1 3 5 0]
 [3 1 2 2]]
   
- What is the rank and trace of A?
    - R:// The rank is 3 and the trace is 7
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
          the euclidean distance. The calculated distance is: 34933.885
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
    - R:// Based on the visual inspection of the reconstructed images with different singular values used,
          I can say tha the images is appropriately reproduced from 25 or more singular values
- How would you quantify how different your photo and the approximation are?
    - R:// We can determine ghow different my phot is against the aproximated reconstructed image calculating
            the euclidean distance between both images. In this case the distance is 1336.908
----------------------------------------------------------------------
- // To see task 4 development details look and run the task4.py script

  
----------------------------------------------------------------------
----------------------------------------------------------------------
5. Train a naive logistic regression on raw MNIST images to distinguish between 0s and 8s. We are calling 
this our baseline. What can you tell about the baseline performance?

- R:// After apply the logistic regression, the accuracy of the trained model is  0.9891 wich is a high accuracy.
      The f1 scor for each class is 0.99
----------------------------------------------------------------------
- // To see task 5 development details look and run the task5.py script

----------------------------------------------------------------------
----------------------------------------------------------------------
6. Now, apply dimensionality reduction using all your algorithms to train the model with only 2 features per
image.
- Plot the 2 new features generated by your algorithm
- Does this somehow impact the performance of your model?
    - R:// yes, considering only 2 featrures on each of the 3 techniques (svd, pca, t-sne), the accuracy gets reduced a
      little bit, but the performance (computing time) is significantly improved except the t-sne wich takes to long
      iterating over the differents steps used in the algorithm

      The accuracy of the models are as next:
      - SVD with 2 features. Accuracy: 0.9541
      - SVD with 4 features. Accuracy: 0.9778
      - PCA
      - 
----------------------------------------------------------------------
- // To see task 6 development details look and run the task6.py script


----------------------------------------------------------------------
----------------------------------------------------------------------
7. Repeat the process above but now using the built-in algorithms in the Scikit-Learn library.
- How different are these results from those of your implementation? Why?
    - R:// the accuracy obtained are practically the same in the both svd and pca, but the t-sne results are quite different because
    - of the different approach used by me vs the built in algorithm
----------------------------------------------------------------------
- // To see task 7 development details look and run the task7.py script

----------------------------------------------------------------------
----------------------------------------------------------------------
8. What strategies do you know (or can think of) in order to make PCA more robust? (Bonus points for
implementing them)
[https://nbviewer.org/github/fastai/numerical-linear-algebra/blob/master/nbs/3.%20Background%20Remov
al%20with%20Robust%20PCA.ipynb]
    - R:// In order to improve the PCA performance, you can use some techniques like:
              - Eliminate outliers: Let the data not beig affected by the outliers
              - Normalize the data: It helps to have the data in the same scale
              - Apply a crossvalidation: To improve the components selection
----------------------------------------------------------------------



----------------------------------------------------------------------
----------------------------------------------------------------------
9. What are the underlying mathematical principles behind UMAP? What is it useful for?
    -R:// UMAP is a dimensionality reduction technique and it consists in the mapping the data from a high dimension  space to a los dimension space, but
           preserving the orogonal data relationship structure. It means that close data points must be close in the reduce space, and the separated ones
           must continue being separated.


   
----------------------------------------------------------------------


----------------------------------------------------------------------
----------------------------------------------------------------------
10. What are the underlying mathematical principles behind LDA? What is it useful for?
----------------------------------------------------------------------


----------------------------------------------------------------------
----------------------------------------------------------------------
11. Use your unsupervised Python package as a basis to build an HTTP server that receives a record as input
and returns the class of the image. Suggestions: MNIST digit classifier, Iris classifier..

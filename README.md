## Project and Data

This deep learning project utilizes data in the field of high-energy particle physics. The data was obtained from the University of California Irvine (UCI) machine learning repository, maintained by the Center for Machine Learning and Intelligent Systems. Here, we are examining data that was generated via Monte Carlo simulations (That is, random sampling of inputs are utilized to create a probabilistic system of the data involved. This data is set within reasons given the background knowledge of the topic. Thus, probabilistic data can be generated without having access to particle accelerators.) 

Particle accelerators, such as the Large Hadron Collider (LHC) collide particles together at near the speed of light to create high density particles. Baldi, Sadowski, and Whiteson note that accelerators such as the LHC can produce $10^{11}$ collisions hourly. Further, of these hundreds of billions of collisions, 300 or so generate in producing Higgs boson (a particle first discovered in 2012 by the LHC and the main scientific discovery to have occurred so far, from the accelerator). Due to the low probability of collisions resulting in Higgs bosons, it is crucial to classifier when a collision has resulted in the boson. To do this, data indicating background noise and true signals of the boson themselves need to be determined and used in the classification. The massive amounts of data generated from the collisions prove themselves as good candidates for deep learning to train on the data presented.

In the vein of large amounts of data from particle accelerators, this simulated dataset contains 11,000,000 instances of 28 features and a class associated with each of these instances. The 28 features are all numeric values and are of float64 type. The actual features themselves and their meaning as the particles at hand are vastly beyond my lay-person's knowledge of physics. However, in UCI's repository, they state that 21 of the 28 features are low-level features and that 7 are high-level. Further, the classes of the data are binary and are labeled as 0 and 1. 

In the paper itself, 500,000 instances are utilized as the test set and I will mirror this approach. 


Initial academic paper for data: https://arxiv.org/abs/1402.4735

Repository for the data: http://archive.ics.uci.edu/ml/datasets/HIGGS

Monte Carlo: https://news.mit.edu/2010/exp-monte-carlo-0517

Particle Accelerators: https://www.home.cern/science/accelerators/large-hadron-collider

## EDA

To begin, I got the column names for each feature and then changed the column initially provided to the correct column name. After that, I wanted to get an overall view of the data. Utilizing the info of the df, we see every data point is of float 64. Further, we also see the magnitude of this dataset as its memory usage is 2.4 GB which is high for numeric values, but expected given the number of instances. Then, I utilized the detail method of the df. With this, I was largely checking for outliers, but it also provides a quick picture of the mean, sd and other summary statistics. From this, I do not immediately see any outliers. Next, I looked to see if there were any na's in the dataframe and there were none present. 

Then, I wanted to check if the data was imbalanced among the two classes. We found that a majority of the data were of the class 1, but this imbalance was slight and thus, measures were not taken to remove it. After this, I wanted to visualize how the numerical data stacked up among its features. Thus, I utilized altair to create a chart of the median values for each median in the training and test set. Further, I added error bars on these medians which accounted for += 1.96 sd of the median. The chart was made interactive, so that one can hover their mouse over a feature and this will highlight that feature on both charts for training and test data. 

After this, I calculated correlations for all features in training and test sets and plotted these correlations as a heatmap with seaborn. In this, I set a threshold that the correlation must have an absolute value of $>=0.1$. From the heatmap, we can see we do not have a lot of correlation in the data with most values being quite below 0.5 and only a couple variables appear to be heavily correlated. 

Lastly, I created a histogram for each of the features and we see that our data is very non-normal (just with a visual test, statistical tests were not ran to test this) and that only a few columns appear normal. I was surprised at this given the large amount of instances.

## Preprocessing

Due to the large number of instances of the data and the fact I am using my personal Macbook Air, I sought to improve model efficiency. 

I first intended to do this by scaling the data with sklearn's MinMaxScaler. This normalizes the data by placing all values in a range of $[0-1]$. This is achieved by taking each value $x$ for a feature and:
$\frac{x - x_{min}} {x_{max} - x_{min}}$. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*This proved to be a mistake in my logic. While I was aware normalized data is not required for neural network. I was surprised to find my test accuracy dipped much lower than training and validation accuracy. This occurred when I applies the same normalization to the test data. When this was done, training and validation accuracy was ~0.78 and test accuracy was ~0.65. This type of drop off was not found in my training models when data was not normalized. Hence, I chose to bypass normalization.*

Additionally, I converted dataframe into a TensorFlow dataset. Doing this allowed me to better utilize the limited GPU I have at my disposal. This allows the next iterations of the dataset (i.e. shuffling for each iteration) to begin work while the current iteration of the dataset is being created (achieved through the use of dataset.prefetch). Additionally, when creating my datasets, I transferred the float values from float64 to float32. By switching from 64 to 32 bits per data point, the memory is vastly improved in my model's speed performance. I anticipated a decrease in model performance; however, when training on both float64 and float32, there was only a marginal difference in accuracy. Then, with my dataset, I got batches of it in size 10,000 and the dataset was then shuffled through each iteration of the model. I also set a cache for the elements of the dataset. This means that in the first iteration, the elements of dataset will be saved to a cache and these elements will drawn from the cache in each subsequent epoch. Again, this is a method for improving model efficiency.

Lastly, I split the batched dataset into training and validation datasets which were 0.8 and 0.2 of the initial dataset size, respectively.




Sources: 

https://www.educative.io/answers/standardization-vs-min-max-normalization

https://ui.adsabs.harvard.edu/abs/2012arXiv1203.3838C/abstract

https://www.tensorflow.org/guide/data_performance

https://medium.com/when-i-work-data/converting-a-pandas-dataframe-into-a-tensorflow-dataset-752f3783c168

## Modeling

Given the type of data I was working with, I knew I would largely be working with Keras Dense layers and a single output layer. However, there was still much to determine in terms of model architecture. Firstly, in the academic paper, I saw they conducted their model architecture with a tanh activation on all hidden layers. I was surprised at this and I initially did not follow their own decision. Thus, I began by using relu as the activation function on all my hidden layers as that is my standard choice. However, after a few iterations of my model I decided to follow the author's methodology and I too found it to optimize training accuracy. Lastly, in terms of activation function, I utilized tanh as my output function. Traditionally, I have utilized sigmoid as my output for binary classification; however, I chose to utilize tanh here given the fact that we do have negative data values and I thought tanh may help given its range of -1 to 1. Then, for the model architecture, I ran quite a few iterations to determine the number of layers needed. This is one of the largest, if not the largest dataset I have worked with; hence, I was unsure how complex the model needed to be to avoid any potential overfitting, but still have enough to learn and train. Eventually, I settle on 14 hidden layers and an output layer (with also an input layer). There was no theoretical explanation for this final number, but simply trial and error until I eventually was consistently getting an accuracy I deemed acceptable. 

Then, in terms of compiling the model, I utilized the Adam optimizer with an initial learning rate of 0.001. However, learning rate scheduler was utilized and after the first epoch, the learning rate was decreased exponentially each subsequent epoch (to provide more in depth learning and slower weight updates as the model continues to run). Then, the loss of the model was measured in binary crossentropy, given this was a binary classification problem. Lastly, the metrics the model was compiled on were accuracy, area under the ROC curve and recall. I chose to add other metrics given the desire to avoid false negatives in the product, but I also typically utilize other metrics besides accuracy for a more robust model. Area under the ROC curve looks at the area under the curve when plotted on the true positive and false positive rate of the data, with the desired AUC to be as close to 1 as possible. Additionally, recall is a measure of the true positive rate and the false negative rate and this value should also be as close to possible to 1.

After the model was compiled, it needed to be fitted to the training data. This was done with the training data being analyzed with a batch size of 1000 (another number derived through trial and error). Then, early stopping was created with a patience of 1. That is there must be one full epoch of no decreasing loss before stopping. Additionally, my final number of epochs was 45. The authors of the initial paper utilized 200 plus epochs, but my computer did not allow for this thorough of a computation. Further, my early stopping tended to stop the model at the end of the 30's epochs or in the early 40's, so I am not sure if my model as it was created would benefit from more epochs. After the model completed its fitting, the training and validation accuracy were charted on a plot and the training and validation loss were also charted on their own plot. This helps to serve as a visual indicator of the model's performance throughout the epochs.

## Predictions and Results

To get my predictions, I take the trained model and predict it on the test dataframe I created in the beginning. Then, for each value in predictions, if prediction is $>=0.5$, then a 1 is predicted and else a 0 is predicted. Next, I calculated accuracy, recall, precision and f1 score through sklearn packages and these values were printed. 

## Conclusion

In conclusion, this project looked at simulated data of a particle accelerator. Its objective was to build a classifier which could identify when Higgs bosons were produced from the collisions formed in the accelerator or not. Accelerators such as Large Hadron from CERN can perform $10^{11}$ collisions in an hour with only approximately 300 producing bosons. Thus, given the low probability of occurrences an adequate classifier is vastly needed.

The data provided by the UCI Machine Learning repository included 11,000,000 instances of 28 features and a target class. Exploratory data analysis was performed, with included data visualizations, data was preprocessed utilizing a tensorflow dataset which vastly improved speed of modeling, and a Keras sequential model was built and trained on a split of the train/test split of the data, the the model was used to classify the test split of the initial data and metrics of accuracy, recall, precision, and f1 score were measured.

The results from the finalized product were:

Accuracy: 0.777
Recall: 0.809
Precision: 0.779
F1 Score: 0.794


Github repo: https://github.com/skembr01/Higgs_NN_Classifier

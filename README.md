# Process-Faut-Diagnosis-A-Supervised-Machine-Learning-Approach

Introduction:

Welcome to this video series on machine learning for fault diagnosis. In this series, we will explore various machine learning techniques and algorithms that can be used to diagnose faults in complex systems. We will be using the [Tennessee Eastman dataset](https://www.kaggle.com/datasets/averkij/tennessee-eastman-process-simulation-dataset), which is a widely used benchmark dataset for fault diagnosis. The dataset contains process data for a simulated chemical process with various types of faults.

## [YouTube Playlist](https://www.youtube.com/playlist?list=PLoSULBSCtofc9wzI9xXBMOc_bH08PUqQ5)
<img src="https://github.com/mohan696matlab/Process-Faut-Diagnosis-A-Supervised-Machine-Learning-Approach/blob/main/images/playlist.JPG" width="200" height="400" />

## Requirements
`numpy==1.21.4
pandas==1.3.4
matplotlib==3.4.3
seaborn==0.11.2
scikit-learn==1.1
xgboost==1.5.1
tensorflow==2.7.0
keras==2.7.0`

## Description

### [Part 1: Exploratory Data Analysis (EDA)](https://youtu.be/su3RUtYB69Q)
In this video, we will explore the Tennessee Eastman dataset and perform some exploratory data analysis (EDA) to gain insights into the data. We will visualize the data, perform statistical analyses, and identify patterns and anomalies in the data.

### [Part 2: Training Various Machine Learning Algorithms with Hyperparameter Tuning](https://youtu.be/__x2kIR23TI)
In this video, we will train various machine learning algorithms on the Tennessee Eastman dataset. We will use techniques like cross-validation and hyperparameter tuning to optimize the performance of these algorithms. We will evaluate the accuracy of each algorithm and compare their performance.

### [Part 3: Neural Networks for Fault Diagnosis](https://youtu.be/Goh_kZewwhw)
In this video, we will explore the use of neural networks for fault diagnosis. We will start by discussing the basics of neural networks, including feedforward and recurrent neural networks. We will then implement a neural network on the Tennessee Eastman dataset and evaluate its performance.

### [Part 4: Convolutional Neural Networks (CNNs) for Fault Diagnosis](https://youtu.be/eC2OWRczxRU)
In this video, we will focus on time series analysis using convolutional neural networks (CNNs). We will start by discussing the basics of CNNs and how they can be used for time series analysis. We will then implement a CNN on the Tennessee Eastman dataset and evaluate its performance.

### [Part 5: Long Short-Term Memory (LSTM) Networks for Fault Diagnosis](https://youtu.be/W715Ix3Khrw)
In this video, we will explore the use of long short-term memory (LSTM) networks for time series analysis. We will start by discussing the basics of LSTM networks and how they can be used for time series analysis. We will then implement an LSTM network on the Tennessee Eastman dataset and evaluate its performance.

### [Part 6: ANN + Random Forest based Fault Diagnosis](https://youtu.be/KdwcysZB5GU)
This video will demonstrate use of a hybrid machine learning method by combining the Neural Network and random forest. First an ANN is trained on the dataset and then the embeddings of generated from the learned are used as input to a Random Forest for classification.

### Metrics:
In this series, we will use accuracy as the primary metric for evaluating the performance of the different machine learning algorithms. We will update the table as we evaluate the performance of other algorithms in the subsequent videos.


Average accuracy score obtained for each method, excluding fault No. 9 and 15 (**No feature were Dropped, all 52 sensor measurements were used**)
| Method                                    |Accuracy  |
|-----------------------------------------  |----------|
| XG Boost                                  |  0.887   |
| Neural Network                            |  0.943   |
| 1D CNN-Timeserise                         |  0.892   |
| LSTM-Timeserise                           |  0.924   |
| ANN+RandomForest                          |  0.936   |

## Other Sources
- [Unsupervise Fault Detection on TEP dataset, YouTube Playlist](https://www.youtube.com/playlist?list=PLoSULBSCtoffIldbr898SDp5gIqo8XL-t)

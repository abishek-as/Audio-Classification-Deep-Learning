# About the Project

Kaggle Notebook link - <https://www.kaggle.com/abishekas11/audio-classification-using-deep-learning>.<br>
The demo of the project is explained here - <https://youtu.be/hJvr1dyiOxM>.

## To run this project you need

- python3
- pip

## Steps for running this project

- git clone this project and extract it and open a terminal in that folder itself.
- Create a virtual environment by running the following command
  - ```python3 -m venv audio-venv``` (audio-venv is the name of the virtual environment)
- Activate the virtual environment by running the following command
  - ```source audio-venv/bin/activate```
- Now run the following command
  - ```pip install -r requirements.txt```
- Now run the following command to start the Django server
  - ```python3 manage.py run server```
- To stop the server, press Ctrl+Shift+C or Ctrl+Alt+C

## Abstract

Audio classification or sound classification can be referred to as the process of analysing audio recordings. This amazing technique has multiple applications in the fields of AI and data science.
In this project, we will explore audio classification using deep learning concepts involving algorithms like Artificial Neural Network (ANN), 1D Convolutional Neural Network (CNN1D), and CNN2D. The dataset contains 8732 labelled sound excerpts (=4s) of urban sounds from ten categories: air for audio prediction, car horns, children playing, dog barking, drilling, engine idling, gunshots, jackhammers, sirens, and street music are used for audio prediction. Before we develop models, we do some basic data preprocessing and feature extraction on audio signals. As a result, each model is compared in terms of accuracy, training time, and prediction time. This is explained by model deployment, where users are allowed to load a desired sound output for each model being deployed successfully, which will be discussed in detail.

## Introduction

Sound classification is one of the most widely used applications in audio deep learning. It involves learning to classify sounds and predicting the category of that sound. This type of problem can be applied to many practical scenarios, e.g., classifying music clips to identify the genre of the music, or classifying short utterances by a set of speakers to identify the speaker based on the voice. Our project involves a comparison of some deep learning models and successfully deploying them using Django. For comparison and deployment, we have selected three of the most widely used deep learning models like ANN, CNN1D, and CNN2D. Before building and deploying the models, we will do some preprocessing and feature extraction. A detailed comparison study of accuracy, training and prediction time, and finding the best model amongst the models is made at the end of the document. As our aim is to help deaf people know their surroundings, we have deployed our model in use. Here people can load the audio files (.wav) and, once they submit them, the audio sound is printed as the outcome for each model, thus achieving our goal for the project.

## Objective and goal of the project

We always thought we could develop a useful application by applying deep learning concepts that help deaf people hear what is happening around them. We are all blessed with good hearing, but some are unlucky. So, we thought, why don't we develop some software useful for those people to upload the audio files and see a report which gives a detailed classification of the audio file. This not only helps them recognise what is going on around them, but it also helps them connect with the world in the same way that we do. Our Goal is to develop a very useful application for deaf people that helps them know the voices around them and provides a summary of those voices.

## Problem Statement

Nowadays, deep learning is an emerging topic for upcoming IT professionals. Deep learning is mostly used in audio or image processing projects. So we thought of doing audio classification using deep learning models as our project. Audio classification or sound classification can be referred to as the process of analysing audio recordings. This amazing technique has multiple applications in the field of AI and data science, such as chatbots, automated voice translators, virtual assistants, music genre identification, and text-to-speech applications. Many deaf people suffer a lot as they cannot interact with the outside world and their relationship with the environment is not like we all have. As a group of software engineering students, this problem made us think of deploying a useful solution for the above-mentioned problem. So we came up with the idea of deploying a web application allowing deaf people/users to load audio files and get to know what all the sounds they are surrounded with using deep learning models. Here, we are also doing a comparison of some deep learning models and successfully deploying them using Django. For comparison and deployment, we have selected three of the most widely used deep learning models, like ANN, CNN1D, and CNN2D. Before building and deploying the models, we will do some preprocessing and feature extraction. A detailed comparison study of accuracy, training and prediction time, and finding the best model amongst the models is made at the end of the document. With the successful deployment of the web application, deaf people can get to know their surrounding sounds, which satisfies our main goal of the project.

## Literature Survey

1. [An Analysis of Audio Classification Techniques using Deep Learning Architectures](https://sci-hub.se/https://doi.org/10.1109/ICICT50816.2021.9358774)., - They used CNN and RNN neural network models to accomplish the classification in this study, and a comparison of the accuracy of each model is done. They began with standard audio preprocessing techniques such as normalization, rapid Fourier transformation, and short-time Fourier transformation, all of which were conceptually and thoroughly documented in the study. For training and testing, the datasets used were UrbanSound8K, ESC-50, and FSDKaggle2018. In total, these datasets contain nearly 7000 sound excerpts. During each training cycle, all of the samples (audio) in the dataset were chosen at random. They've also looked at the CF model and the CFClean model to see which produces better outcomes. 0.0001 is the chosen learning rate. And now, as previously stated, the CNN and RNN models are used, with a comparative analysis of accuracies and loss percentages for both models provided in a table format for easier analysis. Finally, CNN outperformed RNN with a maximum accuracy of 94.52 percent.
2. [Automatic Sound Classification Using Deep Learning Networks](http://dspace.lpu.in:8080/jspui/bitstream/123456789/3157/1/11602389_11_29_2017%203_44_29%20PM_Complete%20File.pdf)., - They propose a model that uses a Tensor Deep Stacking Network to classify sounds in this paper (T-DSN). To achieve the low error rate in classification, an HMM and SoftMax layer will be utilized in addition to the Tensor deep stacking network. The fundamental concept of deep learning has been explained. The Tensor Deep Stacking Network (T-DSN) is a Deep Stacking Network extension (DSN). To obtain greater precision than the preceding layer, the original input vector is also concatenated with the intermediate output of the hidden layer. A deep stacking network differs from previous architectures in that it uses the mean square error between the current module's forecast and the ultimate prediction value rather than the gradient descent method. With this new strategy, the mistake rate will be minimized. Their main goals in this research are to compare the performance of a traditional neural network to that of deep learning architecture. This goal will justify the usage of deep architectures for massive data sets. Tensor-Deep Stacking Network with Hidden Markov Model and Nonlinear Activation for Sound Classification. This goal will be accomplished by merging a classic sound classification model with a new tensor deep stacking network technique. This research report enlightens us on the T-DSN methodology and how it is used in data.
3. [Unsupervised feature learning for audio classification using convolutional deep belief networks](http://citeseerx.ist.psu.edu/viewdoc/downloaddoi=10.1.1.154.380&rep=rep1&type=pdf)., - In this paper, they have applied Convolutional Deep Belief Networks (CDBN) to evaluate several audio classification tasks. For the application purpose, we convert time-domain signals that we got into spectrograms then we apply PCA whitening to create a lower-dimensional representation, and that the data to apply CDBN contains n number of PCA components of one-dimensional vectors of length(n). Feature learning for unsupervised data (training, visualization, Phonemes, and the CDBN features application), speech data identification, and discovering the accuracy for each data type such as a speaker, phone, and so on are all processes involved. Finally, we use music data in this paper (identification, music artist classification, and accuracy analysis using the CDBN model). All of the accurate information has been meticulously tabulated.
4. [Audio Based Bird Species Identification using Deep Learning Techniques](https://infoscience.epfl.ch/record/229232/files/16090547.pdf)., - The CNN model is used in this article to predict bird species identification. Audio processing, data augmentation, bird species recognition, and acoustic classification are among the services they provide. The sound file is separated into two parts for preprocessing: a signal part with audible bird audio and a noise part with no audio. The spectrogram is divided into equal-sized chunks for both portions, with each chunk serving as a separate sample for CNN. A batch size of 16 training samples per iteration was also employed during training, which was detailed in detail using a figure that clearly describes each step for greater understanding. This paper aids us in comprehending each phase.
5. [Efficient Pre-Processing of Audio and Video signal dataset for building an efficient Automatic Speech Recognition System](https://www.acadpubl.eu/hub/2018-119-16/1/189.pdf)., - They preprocess the dataset, which includes video and audio files (.wav files), then use algorithms to create an artificial speech recognition system in this article. This article is mostly used to prepare audio files for subsequent processing. Some audio signal features are retrieved and transformed for use in the model. .wav files are read using Python modules, and a 1-d NumPy array with the sample rate is returned. The 13 most important MFCC features for an audio signal are extracted using the mfcc function in Python. To ensure that the model generalizes well to real data, we introduced background sounds to the audio files. The process is then performed twice more, with each noisy file generated being unique.
6. [CNNs for Audio Classification](https://towardsdatascience.com/cnns-for-audio-classification-6244954665ab)., - In this article, a Kaggle dataset is downloaded and used to train CNN. There are a lot of audio files in the dataset you choose. We first import all libraries, such as Libros. CNN anticipates a grayscale image and a three-channel color image (RGB). Now we can start modeling the data, normalize it, and cast it into a NumPy array. The CNN model is now generated, and the model is then evaluated. This tutorial will assist us in implementing the CNN algorithm successfully. Similarly, we obtained references for ANN, RNN, and other topics from study material websites such as geekforgeeks, tutorials point, and so on. We can successfully learn and implement the algorithms using these sources.

## Feasibility Study

A feasibility study is an analysis that takes all of a project's important factors into account—including economic, technical, legal, and scheduling problems—to ascertain the likelihood of completing the project successfully. Project directors use feasibility studies to distinguish the pros and cons of undertaking a project before they invest a lot of time and money into it.

### Technical Feasibility

#### Gathering dataset online

The first technical problem we face is gathering the dataset. Because we are using .wav files, it took us longer to collect audio files comprising various voices. Though it took longer than expected, the subsequent process was relatively simple.

#### Memory used

Because the dataset we utilized was 6.60GB in size, training and testing each model required more memory and time. It nearly took us an hour to complete the period for each model. We didn't utilize much RAM because we used Google Collab.

#### Deploying the model

Initially, we considered utilizing any model and uploading the .wav file to acquire the appropriate outcome, as our major goal is to assist deaf individuals in recording and identifying any noises present. However, because we are also performing comparative analysis, we decided to display the outcome predicted by each of the models that we employed for better performance. This work was technically achievable as well.

#### Accessibility

The hardware required for this project is straightforward and widely available. All that is required is any type of optical instrument and a computer to interpret and assess the acquired sights.

### Economic Feasibility

The question of whether the model can be proven to be economically feasible is one of the most significant barriers and misconceptions of any new technology. The following study explains the project's financial viability. This is a low-cost project that may be used by anyone who has a touch phone or a laptop. There aren't many hardware and software costs to worry about. The only thing to consider is the cost of maintenance. Following the successful deployment of our project, we must regularly evaluate the operation of the four projects; if there are more users, we must upgrade our project so that more users can benefit.

### Social Feasibility

Deaf and blind people will be unaware of how their surroundings treat them. They will gain completely from our project. All they have to do is document their surroundings and be aware of what is going on around them. This will make them happy, and their worry of not knowing what is going on around them will be gone.

### Environmental Feasibility

An Environmental Feasibility Study evaluates the environmental and social viability of a planned development, identifying potential challenges and dangers to the successful completion of the proposed development. Solutions and mitigating strategies are being researched. The goal of an Environmental Due Diligence Report is to assess potential risks and liabilities related to environmental and health and safety issues such as land contamination before entering into any contractual agreements. This is critical since these risks could result in financial liabilities for the parties concerned.

#### Environment and Health

This project is extremely beneficial to deaf individuals. The recording of their surroundings is critical to the success of our initiative. There will be no harm done to the environment's health because our project is entirely technical, and its safety will not be jeopardized by any factor.

### Political Feasibility

There are many new solutions for physically challenged persons on the horizon, and we wanted one of our efforts to benefit the public. According to our hypothesis, this project will benefit deaf individuals and bring them delight, thereby satisfying themselves as well as the project purpose. This project may take some time to complete, but it will never fail.

## System Design

### About the Dataset

Dataset taken from - [https://urbansounddataset.weebly.com/urbansound8k.html]

#### Description

This dataset contains 8732 labeled sound excerpts (=4s) of urban sounds from ten categories: air conditioner, car horn, children playing, dog bark, drilling, engine idling, gun shot, jackhammer, siren, and street music. The classes are based on the taxonomy of urban sounds. All excerpts are from field recordings that have been uploaded to www.freesound.org. The files are pre-sorted into ten folds (folders named fold1-fold10) to aid in reproducing and comparing the results of the automatic classification.

#### Audio Files Included

8732 WAV audio files of urban sounds (as described above).

#### Metadata Files Included

- **UrbanSound8k.csv** : This file contains meta-data for each audio file in the dataset. This includes the following:
- The name of the audio file. The name takes the following format: (fsID)-(classID)-(occurrenceID)-(sliceID).wav, where: (fsID) = the Freesound ID of the recording from which this excerpt (slice) is taken, (classID) = a numeric identifier of the sound class (see description of classID below for further details), (occurrenceID) = a numeric identifier to distinguish different occurrences of the sound within the original recording, (sliceID) = a numeric identifier to distinguish different slices taken from the same occurrence
- fsid : The Freesound ID of the recording from which this excerpt (slice) is taken
- start : The start time of the slice in the original Freesound recording
- end: The end time of slice in the original Freesound recording
- salience: A (subjective) salience rating of the sound. 1 = foreground, 2 = background.
- fold: The fold number (1-10) to which this file has been allocated.
- classID : A numeric identifier of the sound class:  0 = air_conditioner, 1 = car_horn, 2 = children_playing, 3 = dog_bark, 4 = drilling, 5 = engine_idling, 6 = gun_shot, 7 = jackhammer, 8 = siren, 9 = street_music
- class: The class name: air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music.

### Algorithm Used

#### Artificial Neural Network (ANN)

Artificial neural networks (ANNs) are made up of node layers, each of which has an input layer, one or more hidden layers, and an output layer. Each node, or artificial neuron, is linked to another and has its own weight and threshold. If the output of any individual node exceeds the specified threshold value, that node is activated and begins sending data to the network's next layer. Otherwise, no data is passed to the next network layer.

#### Convolutional Neural Network (CNN)

A convolutional neural network (CNN, or ConvNet) is a type of artificial neural network used to interpret visual imagery in deep learning. Based on the shared-weight architecture of the convolution kernels or filters that slide along input features and give translation equivariant responses known as feature maps, they are also known as shift invariant or space invariant artificial neural networks (SIANN). Surprisingly, most convolutional neural networks are only equivariant under translation, rather than invariant. Image and video recognition, recommender systems, image classification, image segmentation, medical image analysis, natural language processing, brain-computer interfaces, and financial time series are just a few of the areas where they can be used.

##### CNN2D

The traditional deep CNNs discussed in the preceding section are only designed to work with 2D data like photos and movies. This is why they're referred to as "2D CNNs" so frequently. 2D CNN, kernel moves in 2 directions. Input and output data of 2D CNN is 3 dimensional. Mostly used on Image data. This is the conventional Convolution Neural Network, which was introduced in the Lenet-5 architecture for the first time. On Image data, Conv2D is commonly used. Because the kernel slides along two dimensions on the data, it is called 2 dimensional CNN. The main benefit of employing CNN is that it can extract spatial properties from data using its kernel, which is something that other networks can't achieve. CNN, for example, can detect edges, colour distribution, and other spatial aspects in an image, making these networks particularly robust in image classification and other data with spatial qualities.

##### CNN1D

A modified form of 2D CNNs known as 1D Convolutional Neural Networks (1D CNNs) has recently been developed as an alternative. In dealing with 1D signals, these research have proven that 1D CNNs are beneficial and consequently preferable to their 2D counterparts in some situations. 1D CNN, kernel moves in 1 direction. Input and output data of 1D CNN is 2 dimensional. Mostly used on Time-Series data. 1D CNN can perform activity recognition task from accelerometer data, such as if the person is standing, walking, jumping etc. This data has 2 dimensions. The first dimension is time-steps and other is the values of the acceleration in 3 axes. Similarly, 1D CNNs are also used on audio and text data since we can also represent the sound and texts as a time series data. Conv1D is widely applied on sensory data, and accelerometer data is one of it.

### Architecture Diagram

#### ANN

![Model1](/images/Model1.png)

#### CNN1D

![Model2](/images/Model2.png)

#### CNN2D

![Model3](/images/Model3.png)

### Proposed Methodology

#### Importing the Dataset, Data Preprocessing

We obtained the dataset from UrbanSound8K, which contains over 8500 data files containing various audios such as a baby crying, birds sound, dog bark, and many others in the form of.wav files. It is divided into ten folders, indicating that the dataset has ten classes as described in the dataset section. Libraries that are required are added, as well as Librosa for feature extraction.

#### Feature Extraction and Database Building

We use the data we obtained by using librosa because we require data in numeric format. We use Mel-Frequency Cepstral Coefficients (MFCC) to extract independent data, which summarizes the frequency distribution across the window size, allowing us to analyze both the frequency and time characteristics of the sound. We can identify features for classification using these audio representations. We defined the features extractor function and passed in the path to the audio file as a parameter, after which we will extract the audio features using librosa. The feature_extractor function is applied to all rows, and the results are stored in a dataframe with features and class columns for further calculations. The feature_extractor function is attached in appendix below.

#### Building, Training, Compiling ANN, CNN1D \& CNN2D Models

We save the dataframe's feature and class columns to x and y arrays, respectively. The y array is then converted to categorical values using the Labelencoder() method. The data was then divided into test and training sets. Then, based on the architecture diagrams, we design, compile, and fit the ANN, CNN1D, and CNN2D models, and save the findings for future visualization.

#### Predicting the test audio on all the Models

We Defined a function which will extract the features from the audio which is given in the parameter , and the features will be converted to proper input shapes for ANN, CNN1D and CNN2D modles. The output will give the class label. The ANN_print_prediction, CNN1D_print_prediction and CNN2D_print_prediction functions is attached in appendix bellow.

#### Deployment of Models

We deploy the model in a web application using the Django web framework. Now the test data is applied to the model and the output is displayed.

## Implementation of System

### Homepage

![Homepage](/images/21.png)

### Select the sample audio for testing

![Select the sample audio for testing](/images/22.png)

### Uploading the sample audio file

![Uploading the sample audio file](/images/23.png)

### Result page

![Result page](/images/24.png)

## Results & Discussion

### Comparing Accuracy of All Modules

ANN has highest accuracy of all modules.
![Comparing Accuracy of All Modules](/images/12.png)

### Comparing Training time of All Modules

CNN1D training takes more time of all modules.
![Comparing Training time of All Modules](/images/13.png)

### Comparing Prediction time of All Modules

CNN2D prediction takes less time of all modules.
![Comparing Prediction time of All Modules](/images/14.png)

### Loss per Epochs and Accuracy per Epochs for ANN Model

![Loss per Epochs](/images/15.png)
![Accuracy per Epochs](/images/16.png)

### Loss per Epochs and Accuracy per Epochs for CNN1D Model

![Loss per Epochs](/images/17.png)
![Accuracy per Epochs](/images/18.png)

### Loss per Epochs and Accuracy per Epochs for CNN2D Model

![Loss per Epochs](/images/19.png)
![Accuracy per Epochs](/images/20.png)

## Performance Analysis

In our project, we got audio signals and sample rates for the selected audio files, which was a difficult challenge for us. We used numerous references for each algorithm and conducted a comparison analysis of three algorithms: ANN, CNN1D, and CNN2D. Using all the techniques, we tested the algorithm with various audio files and obtained the projected value as expected. We displayed all the comparisons we made, including accuracies, training time, and prediction time for each model, to provide a visual depiction. In terms of accuracy, ANN has a 94.79\% accuracy rate, CNN1D has a 93.68\% accuracy rate, and CNN2D has an 89.93\% accuracy rate, therefore we can conclude that ANN has the highest accuracy rate.

## Conclusion & Future Work

There are so many research papers that tells us how an algorithm works and how to predict any model or algorithm. Those references helped us a lot in achieving our project goal. As a result, we have successfully done a comparative analysis for accuracy rate, training time and prediction time. We have also deployed our project using Django and was a very challenging task. In future work, we can also do comparative analysis on many models to get a better understanding of any model. Also, we can deploy an updated model that allows users to record their surroundings using a mic and get the desired output. This way, users can easily use the model deployment anywhere without any restrictions like file format not supported or large file size and so on.

# Precision-Oncology-Transforming-Cancer-Treatment
> Breast cancer the most common cancer among women worldwide accounting for 25 percent of all cancer cases and affected 2.1 million people in 2015 early diagnosis significantly increases the chances of survival.  
> The key challenge in cancer detection is how to classify tumors into malignant or benign machine learning techniques can dramatically improves the accuracy of diagnosis
> https://www.mskcc.org/cancer-care/types/breast/diagnosis/stages-breast

Research indicates that most experienced physicians can diagnose cancer with 79 percent accuracy while 91 percent correct diagnosis is achieved using machine learning techniques.

_In this case study, our task is to classify tumors into malignant or benign tumors using features of pain from several cell images._

The initial step in diagnosing cancer involves performing a procedure called a needle aspirate, where cells are extracted from a tumor. At this stage, it's not yet clear whether the tumor is malignant or benign. A benign tumor, as shown in the images, is non-cancerous and does not spread to other parts of the body, making it relatively safe. In contrast, a malignant tumor is cancerous and can spread throughout the body.


To effectively address cancer growth, we need to intervene and halt its progression. In the machine learning aspect of this process, we first extract features from images of cancerous cells, aiming to distinguish between malignant and benign tumors. 

We analyze these images to extract various characteristics, such as cell radius, texture, perimeter, area, and smoothness. These features are then input into a machine learning model, which acts like an artificial brain.

The goal is to train the machine to classify images or data as malignant or benign without human intervention. Once the model is trained, it can be used to classify new images efficiently. This trained model can be applied in practical settings to assist with cancer diagnosis, streamlining the classification process as we continue to develop and utilize it.

**# STEP #1: PROBLEM STATEMENT**

-   Determining whether a cancer diagnosis is benign or malignant using multiple observations/features.
-   The analysis involves evaluating 30 different features, such as:
-   
-   `- radius (mean of distances from center to points on the perimeter)`
-   `- texture (standard deviation of gray-scale values) - perimeter`
-   `- area - smoothness (local variation in radius lengths)`
-   `- compactness (perimeter^2 / area - 1.0)`
-   `- concavity (severity of concave portions of the contour)`
-   `- concave points (number of concave portions of the contour)`
-   `- symmetry`
-   `- fractal dimension ("coastline approximation" - 1)`
-   Datasets are linearly separable using all 30 input features
-   Number of Instances: 569
-   Class Distribution: 212 Malignant, 357 Benign
-   Target class:
-   `- Malignant - Benign`

-   # STEP #2: 
# STEP #2: IMPORTING DATA

# import libraries   
import pandas as pd # Import Pandas for data manipulation using dataframes  
import numpy as np # Import Numpy for data statistical analysis   
import matplotlib.pyplot as plt # Import matplotlib for data visualisation  
import seaborn as sns # Statistical data visualization  
# %matplotlib inline  

Import dataset

# Import Cancer data drom the Sklearn library  
from sklearn.datasets import load_breast_cancer  
cancer = load_breast_cancer()

# STEP #3: VISUALIZING THE DATA

sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'] )

![](https://miro.medium.com/max/30/1*k3lpa0s58q9nayfHjsfXuA.png?q=20)

![](https://miro.medium.com/max/957/1*k3lpa0s58q9nayfHjsfXuA.png)

sns.countplot(df_cancer['target'], label = "Count")

![](https://miro.medium.com/max/30/1*FQBQGAJDq3taSIOn_p5LWA.png?q=20)

![](https://miro.medium.com/max/392/1*FQBQGAJDq3taSIOn_p5LWA.png)

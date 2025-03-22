# Cosmic Classifier - IIT Roorkee Cognizance '25

# **Exploratory Data Analysis**  
- **Loading the Dataset** :The CSV file (thermoracleTrain.csv) is loaded into a Pandas DataFrame named df.
- The dataset contains 60000 rows and 11 columns with the Prediction column with values 0 to 9, being the target column.
- **Dataset Statistic Summary** : Mean for all columns (except Prediction) is almost zero indicating the data is scaled. Two categorical columns are present - Magnetic Field Strength and Radiation Levels.

## **Dealing with Missing values in Numerical Columns**
- Rows where the Prediction column is Null are dropped from the dataset.
- The rest of the missing values in numerical columns are filled with the respective medians.
- The value counts of the predicted classes show that the data is fairly balanced.

## **Categorical Columns Labeling and Cleaning**
- The categorical columns- Magnetic Field Strength and Radiation Levels have values Category_1 to Category_20 which are mapped to corresponding values of 1 to 20 (treating them as levels) to numerical columns.
- K-Nearest Neighbours Imputation is used to fill the missing values of the categorical columns.
## **Univariate and Bivariate Feature Analysis**
- Distributions of all features are plotted to check for skewness and understand its overall distributon.
- Pairwise correlation of all the features in plotted in a correlation matrix as a part of bivariate analysis.
- Surface Temperature and Mineral Abundance is seen to have strong positive correlation with the target column.

# **Noise Detection and Removal**
## **Oulier Analysis and Removal**
- Rows with absolute z-scores <3 are retained and else dropped leaving 55778 rows.
## **Noise Detection and Elimination using Ensemble Techniques**
- In the ensemble-based noise detection method, a list of classifiers is chosen to use for predicting class labels on the dataset. These models are not pre-trained; rather, they are instantiatedand then used within a cross-validation framework to generate predictions for each instance.The ensemble method computes how often each instance is misclassified across all the classifiers. The more classifiers that disagree with the true label, the higher the noise score for that instance, suggesting that it might be mislabeled or noisy.
- Top 20 percentile data based on noise scores are removed resulting in 50186 columns remaining.

# **Model Development and Training**  
We developed a custom neural network architecture for the classification task. Details of the same are as follows.

## **Neural Network Architecture for Cosmic Classification**  
This model uses a deep learning approach with **residual connections** to classify cosmic objects.  

### **Residual Block Structure**  
- Each residual block maintains a **skip connection** that allows gradients to flow through the network more efficiently.  
- Uses **L2 regularization** to prevent overfitting.  
- Implements **BatchNormalization** to stabilize learning.  
- Uses **LeakyReLU activation** for better gradient flow.  
- Includes **dropout** for additional regularization.  

### **Network Design**  
- **Input Layer**: Accepts **15 astronomical features**.  
- **Initial Feature Extraction**: **256 neurons** with regularization.  
- **Residual Blocks**: Four blocks with decreasing dimensions (**256 → 192 → 128 → 64**).  
- **Output Layer**: **10 classification categories** using **softmax activation**.  

### **Training Optimizations**  
✅ **SGD Optimizer**: Uses **momentum** and **Nesterov acceleration**.  
✅ **Adaptive Learning Rate**: Reduces learning rate when improvement plateaus.  
✅ **Early Stopping**: Prevents overfitting by monitoring validation accuracy.  
✅ **Data Augmentation**: Applies **random noise** to training data for better generalization.  

This architecture combines modern deep learning techniques like **residual connections, regularization, and normalization** to create a robust classifier for cosmic objects while minimizing overfitting.

## **Model Performance Analysis for Cosmic Classifier**  
This deep learning model demonstrates **strong performance across all classes** with an overall accuracy of **95.85%**.  

### **Performance Metrics**  
📌 **Overall Accuracy**: **95.85%** across all 10 planet classifications  
📌 **Average Precision**: **95.73%**  
📌 **Average Recall**: **95.71%**  
📌 **Average F1-Score**: **95.71%**  

### **Class-Specific Performance**  
- **Best Performing Class**: **Class 1** with **99.00% F1-score** (**99.60% recall, 98.42% precision**)  
- **Lowest Performing Class**: **Class 9** with **91.99% F1-score** (**92.09% recall, 91.89% precision**)  
- **Most Balanced Classes**: **Classes 0, 2, and 6** show nearly equal precision and recall  

### **Error Analysis**  
- **Confusion Patterns**: Class **9** has the most misclassifications, particularly with **Class 4**  
- **False Positives**: Most prevalent in **Classes 8 and 9**  
- **False Negatives**: Most common in **Class 5** (**8% of samples**)  

### **Model Balance**  
✅ The model maintains **consistent performance** across all classes with **minimal variance** in metrics.  
✅ No significant **bias** toward any particular class, despite slight variations in support counts.  
✅ **Weighted averages** closely match **macro averages**, indicating **good class balance**.  

This balanced performance suggests that the model has successfully learned **distinctive features** for each planetary classification while maintaining **generalizability**. 🚀  

## **Key Observations from Model Training**  

### **1. Steady Accuracy Improvement**  
- The **training accuracy** consistently increases, reaching around **95%**.  
- The **validation accuracy** surpasses training accuracy early and stabilizes around **96%**.  

### **2. Loss Reduction Over Time**  
- The **training loss** starts high but steadily decreases, indicating **effective learning**.  
- The **validation loss** also decreases but remains slightly **higher than training loss**.  

### **3. No Major Overfitting**  
- **Validation accuracy** does not drop significantly, and **validation loss** does not increase at later epochs.  
- The **small gap** between training and validation loss suggests **good generalization**.  

### **4. Model Convergence**  
- The model continues improving **without drastic fluctuations**, suggesting **stable training**.  

### **5. Potential for Early Stopping**  
- If **validation loss starts increasing** after more epochs, **early stopping** can be applied to prevent overfitting.    

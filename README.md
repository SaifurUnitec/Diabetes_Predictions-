# Diabetes Prediction with R: A Machine Learning Approach

<img src="D1.jpg" alt="Database Normalization" width="400"/>

# 1. Introduction and Problem Description

Diabetes is a common and highly widespread chronic illness, and it is estimated that over 422 million individuals have diabetes all over the world (Hasan & Yasmin, 2025). Taking care of diabetes well is important to reduce the chance of eye, foot and mouth problems, kidney disease and some types of cancer. Early detection of diabetes is one of the best ways to handle it (Chang et al., 2022). Two methods are used to diagnose diabetes– the first is done by medical professionals and the second is with the help of technology. Early symptoms of diabetes can be hard to find just by conducting manual tests (Chaki et al., 2020). In this context, early identification of diabetes and people at higher risk helps prevent complications and manage the disease well. 

The goal of this research is to recognize the early signs of diabetes and take preventive measures using information from actual patients in the USA. The data for this study is a portion of the Behavioural Risk Factor Surveillance System (BRFSS) data published by Kaggle under the name "Diabetes Health Indicators Dataset". It consists of 253,680 observations. The study focused on 11 variables: 10 feature variables and 1 label variable that showed whether someone had diabetes or not. If the binary target variable is “1”, it means the result is positive for diabetic and if it is “0”, it means the result is negative for non-diabetic. 

Dataset Link: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

In this study, Logistic Regression, Random Forest and Decision Tree are the machine learning algorithms applied by R software. Besides comparing accuracy, other factors such as precision, recall, F1 score and AUC are evaluated to find out which model performs best. This study consists of exploratory data analysis, data preprocessing, result analysis and evaluation, a comparison with other existing studies and a conclusion. 

# 2. ABI Technique Review

It is possible to make real-time decisions with Adaptive Business Intelligence (ABI), as it uses data mining, machine learning and optimization (Michalewicz et al., 2007). ABI systems are different from traditional ones since they keep learning and adjusting to new data which makes them fit for use in areas like healthcare analytics.

In this case, Logistic Regression, Random Forest and Decision Trees will be used as machine learning methods to find patterns and predict diabetes. 

### 2.1 Machine Learning Classifier 
**•	Logistic Regression:**  Logistic regression is used when the dependent variable is binary. It can be used to see how a binary result varies with different independent variables that may be nominal, ordinal, interval or ratio (Hosmer et al., 2013). This type of algorithm uses likelihood and can only produce two results: yes or no, true or false, 0 or 1, high or low. If the probability is less than 0.5, it becomes 0 and if it is more than 0.5, it becomes 1 (Chang et al., 2022).

**•	Random Forest:** Random Forest (RF) combines a number of decision trees to build a forest as an ensemble technique. Each tree in the forest is built by using a random part of the data and selecting a set of features. RF is powerful for both making predictions and for determining which independent variables are most important for predicting a dependent outcome (Mehta & Patnaik, 2021). 

**•	Decision Tree:** Decision Trees help obtain data by applying decision rules to a large number of available datasets (Chang et al., 2022). It is a useful way to show an algorithm that is only made up of conditional control statements. (Bhargav & Sashirekha, 2023). 

# 3. Solution Design and Development
The step-by-step problem solution and development are outlined in Figure 1. It represents the stages of data collection, analysing it and making sense of what it means.

<img src="https://github.com/SaifurUnitec/Diabetes_Predictions_With_R/blob/my-new-branch/D2.jpg?raw=true" alt="Image Description" width="400"/>
Figure 1: Architecture Design

### 3.1 Data Description
The BRFSS collected the US Diabetes Health Indicators Dataset which is now published and shared by Alex Teboul on Kaggle. There are 253,680 observations in the dataset that represent adult respondents from the United States. This study examined 10 feature variables from the dataset such as High Blood Pressure, High Cholesterol, Body Mass Index (BMI), Smoker, Physical Activity, General Health Conditions, Mental Health, Difficulty Walking, Sex and Age. The target variable is set to binary with 1 meaning diabetes and 0 meaning no diabetes.
<img src="https://github.com/SaifurUnitec/Diabetes_Predictions_With_R/blob/my-new-branch/D34.jpg?raw=true" alt="Image Description" width="400"/>

### 3.2 Exploratory Data Analysis
Before using machine learning models, it is essential to use Exploratory Data Analysis (EDA) to discover the structure and patterns in the dataset. 

**3.2.1 Dataset Overview**

<img src="https://github.com/SaifurUnitec/Diabetes_Predictions_With_R/blob/my-new-branch/D3.jpg?raw=true" alt="Image Description" width="400"/>

Table 1: Dataset Structure

Table 1 shows the layout of the diabetes dataset which consists of 253,680 observations and 11 variables. The target variable serves to indicate whether a person is non-diabetic (0) or diabetic (1). Most predictors are health-related and demographic features, including high blood pressure, high cholesterol, smoker, physical activity, difficulty walking and Sex and are either binary or ordinal. BMI, MentHlth (mental health days) and Age are all continuous variables with a high range of values.

**3.2.2 Missing Value**

<img src="https://github.com/SaifurUnitec/Diabetes_Predictions_With_R/blob/my-new-branch/D4.jpg?raw=true" alt="Image Description" width="400"/>

Table 2: Missing Value Check

Table 2 shows that all the variables are clean, no missing values, which suggests the data is in good for early diabetes detection.

**3.2.3 Class Distribution of Target Variable**

<img src="https://github.com/SaifurUnitec/Diabetes_Predictions_With_R/blob/my-new-branch/D5.jpg?raw=true" alt="Image Description" width="400"/>

Table 3: Class Distribution

The data demonstrates that, among all individuals, 86.07% are non-diabetic (label 0) and only 13.93% are diabetic or prediabetic (label 1), meaning the dataset is highly imbalanced (Table 3). If the data is not balanced, machine learning models may favour the majority class (non-diabetic) which can result in not identifying as many people at risk for diabetes. A solution to this is achieved with algorithms like the SMOTE (Synthetic Minority Over-sampling Technique).

**3.2.4 Descriptive Statistics**

The descriptive statistics point out that BMI, HighBP, HighChol, GenHlth and MentHlth are important factors related to diabetes risk. The BMI average is 28.38 and there is a clear right skew (2.12) which points to most people being overweight and a higher risk of diabetes. Moreover, 43% of people have high blood pressure and 42% have high cholesterol, both problems found in people with diabetes. 

**3.2.5 Visualize Distributions (Histograms)**

<img src="https://github.com/SaifurUnitec/Diabetes_Predictions_With_R/blob/my-new-branch/D6.jpg?raw=true" alt="Image Description" width="400"/>

Figure 2: BMI Distribution

<img src="https://github.com/SaifurUnitec/Diabetes_Predictions_With_R/blob/my-new-branch/D7.jpg?raw=true" alt="Image Description" width="400"/>

Figure 3: Mental Health Days

Figure 2 demonstrates that the majority of individuals in this group have a BMI between 25 and 35 which suggests they are mainly overweight. As Figure 3 shows, the data is right-skewed, indicating that a lot of people experience no poor mental health days, but a significant number report struggling with mental health daily for as long as 30 days. Such patterns suggest that both physical and mental wellness play a role in identifying people who are more likely to get diabetes.

**3.2.6 Group Comparisons with Boxplots**

<img src="https://github.com/SaifurUnitec/Diabetes_Predictions_With_R/blob/my-new-branch/D8.jpg?raw=true" alt="Image Description" width="400"/>

Figure 4: BMI by Diabetes Status

In the boxplot (figure 4), the median BMI is generally higher among individuals with diabetes (group 1) than among non-diabetic individuals (group 0). The interquartile range is wider and the BMI range more extreme for the diabetic group which suggests a larger variation among them and a higher likelihood of obesity. 

**3.2.7 Bivariate Bar plots (Categorical Features)**

<img src="https://github.com/SaifurUnitec/Diabetes_Predictions_With_R/blob/my-new-branch/D9.jpg?raw=true" alt="Image Description" width="400"/>

Figure 5: Diabetes vs High Blood Pressure 

<img src="https://github.com/SaifurUnitec/Diabetes_Predictions_With_R/blob/my-new-branch/D10.jpg?raw=true" alt="Image Description" width="400"/>

Figure 6: Diabetes vs Physical Activity

In Figure 5, we can see that more diabetic individuals (label 1) have high blood pressure which means that high blood pressure is a frequent issue and a major risk factor in people with diabetes. According to Figure 6, diabetics are much less likely to be physically active than non-diabetics which might suggest that being inactive can lead to a higher risk of diabetes. As a result, the early diabetes prediction models should take into account both high blood pressure and physical activity.

**3.2.8 Correlation Matrix**

<img src="https://github.com/SaifurUnitec/Diabetes_Predictions_With_R/blob/my-new-branch/D11.jpg?raw=true" alt="Image Description" width="400"/>

Figure 7: Correlation Heatmap of Diabetes Dataset

The correlation heatmap (Figure 7) shows that there is a strong connection between Diabetes_binary and HighBP (0.26), GenHlth (0.29), BMI (0.22) and DiffWalk (0.22), meaning that individuals with these conditions have a higher chance of being diabetic. While there is a slight positive correlation (0.12) between physical activity and diabetes, a negative correlation (−0.12) also exists, meaning that lower levels of physical activity may increase the risk of diabetes. Because of these patterns, including these variables helps create a reliable early diabetes detection model.




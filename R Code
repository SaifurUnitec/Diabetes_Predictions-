# Setup Directory
setwd("D:/R Class")
library(dplyr)

# Install and load required packages
install.packages("tidyverse")
install.packages("ggplot2")
install.packages("psych")
install.packages("corrplot")
install.packages("gridExtra")
install.packages("caret")
install.packages("reshape2")
install.packages("DMwR2")
install.packages("mltools")
install.packages("data.table")
install.packages("ggcorrplot")
install.packages("nnet")         
install.packages("rpart")        
install.packages("rpart.plot")  
install.packages("randomForest")


library(tidyverse)
library(ggplot2)
library(psych)
library(corrplot)
library(gridExtra)
library(caret)
library(reshape2)
library(DMwR2)      
library(mltools)    
library(data.table) 
library(nnet)
library(rpart)
library(rpart.plot)
library(randomForest)


# Load Dataset
data <- read.csv("diabetes_datasetUSA.csv")


############ Exploratory Data Analysis ###########

# View structure and summary
str(data)
summary(data)

# Check for missing values in each column
colSums(is.na(data))

#Class Distribution of Target Variable
table(data$Diabetes_binary)
prop.table(table(data$Diabetes_binary)) * 100

#Descriptive Statistics
describe(data)

#Visualize Distributions (Histograms)
ggplot(data, aes(x = BMI)) +
  geom_histogram(fill = "skyblue", color = "black", bins = 30) +
  labs(title = "BMI Distribution", x = "BMI", y = "Frequency")

ggplot(data, aes(x = MentHlth)) +
  geom_histogram(fill = "salmon", color = "black", bins = 30) +
  labs(title = "Mental Health Days", x = "MentHlth", y = "Frequency")

# Group Comparisons with Boxplots
ggplot(data, aes(x = factor(Diabetes_binary), y = BMI, fill = factor(Diabetes_binary))) +
  geom_boxplot() +
  labs(title = "BMI by Diabetes Status", x = "Diabetes", y = "BMI")

#Bivariate Barplots (Categorical Features)
ggplot(data, aes(x = factor(Diabetes_binary), fill = factor(HighBP))) +
  geom_bar(position = "fill") +
  labs(title = "Diabetes vs High Blood Pressure", x = "Diabetes", fill = "HighBP")

ggplot(data, aes(x = factor(Diabetes_binary), fill = factor(PhysActivity))) +
  geom_bar(position = "fill") +
  labs(title = "Diabetes vs Physical Activity", x = "Diabetes", fill = "PhysActivity")

#Correlation Matrix (Numeric Features)
cor_matrix <- cor(data)
corrplot(cor_matrix, method = "color", type = "upper", tl.col = "black", tl.srt = 45)

#Correlation Heatmap
install.packages("ggplot2")
install.packages("reshape2")
install.packages("RColorBrewer")

# Load libraries
library(ggplot2)
library(reshape2)
library(RColorBrewer)

# Compute correlation matrix
cor_matrix <- round(cor(data), 2)

# Melt the correlation matrix into long format
melted_cor <- melt(cor_matrix)

# Create heatmap using ggplot2
ggplot(data = melted_cor, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradientn(colors = brewer.pal(n = 11, name = "RdYlGn"),
                       limits = c(-1, 1)) +
  geom_text(aes(label = value), color = "black", size = 3) +
  theme_minimal() +
  labs(title = "Correlation Heatmap of Diabetes Dataset",
       x = "Variables", y = "Variables") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1))

################### Data Prepossessing ##################


# Load necessary libraries
install.packages("tidyverse")
library(tidyverse)


# Select relevant variables including the target
selected_vars <- c("Diabetes_binary", "HighBP", "HighChol", "BMI", "Smoker", 
                   "PhysActivity", "GenHlth", "MentHlth", "DiffWalk", "Sex", "Age")

# Subset the data
data_subset <- data[, selected_vars]

# Ensure Diabetes_binary is numeric for correlation
data_subset$Diabetes_binary <- as.numeric(as.character(data_subset$Diabetes_binary))

# Compute correlations with the target
cor_values <- sapply(data_subset[-1], function(x) cor(x, data_subset$Diabetes_binary))

# Convert to data frame for ggplot
cor_df <- data.frame(
  Variable = names(cor_values),
  Correlation = as.numeric(cor_values)
)

# Sort variables by correlation
cor_df <- cor_df %>% arrange(desc(Correlation))

# Create bar plot
ggplot(cor_df, aes(x = reorder(Variable, Correlation), y = Correlation)) +
  geom_col(fill = "purple") +
  coord_flip() +
  labs(title = "Correlation with Diabetes_binary",
       x = "Variable",
       y = "Correlation Coefficient") +
  theme_minimal()

# Convert to data.table for chi-square test
data_chi <- data
data_chi$Diabetes_binary <- as.factor(data_chi$Diabetes_binary)


# Apply chi-square test to each categorical variable

chi_scores <- sapply(names(data_chi)[-which(names(data_chi) == "Diabetes_binary")], function(var){
  tbl <- table(data_chi[[var]], data_chi$Diabetes_binary)
  chisq.test(tbl)$statistic
})

# Sort by importance
chi_scores <- sort(chi_scores, decreasing = TRUE)
chi_scores


#### Data Splitting 

# Convert target to factor
data$Diabetes_binary <- as.factor(data$Diabetes_binary)

# Keep top features based on chi-square and correlation 
selected_features <- c("BMI", "HighBP", "HighChol", "GenHlth", "MentHlth", "Age", "DiffWalk", "PhysActivity")

# Subset data
data_selected <- data[, c(selected_features, "Diabetes_binary")]

# Split into training (80%) and testing (20%) sets
install.packages("caret")
library(caret)

set.seed(123)
split_index <- createDataPartition(data_selected$Diabetes_binary, p = 0.8, list = FALSE)
train_data <- data_selected[split_index, ]
test_data <- data_selected[-split_index, ]

# Count of each class in training and test sets
train_table <- table(train_data$Diabetes_binary)
test_table <- table(test_data$Diabetes_binary)

# Proportions
train_prop <- prop.table(train_table) * 100
test_prop <- prop.table(test_table) * 100

# Combine into a summary data frame
summary_df <- data.frame(
  Dataset = rep(c("Train", "Test"), each = 2),
  Diabetes_binary = rep(c("Non-Diabetic (0)", "Diabetic (1)"), 2),
  Count = c(train_table, test_table),
  Proportion = round(c(train_prop, test_prop), 2)
)

# View the table
print(summary_df)

# Compare number of rows
n_total <- nrow(data_selected)
n_train <- nrow(train_data)
n_test  <- nrow(test_data)

# Show percentage of each split
train_pct <- round((n_train / n_total) * 100, 2)
test_pct  <- round((n_test / n_total) * 100, 2)

data.frame(
  Dataset = c("Train", "Test"),
  Rows = c(n_train, n_test),
  Percentage = c(train_pct, test_pct)
)

#### Apply SMOTE only to training data 

install.packages("smotefamily")  
library(smotefamily)

class_counts <- table(train_data$Diabetes_binary)
majority_count <- max(class_counts)
minority_count <- min(class_counts)

dup_size <- (majority_count - minority_count) / minority_count
dup_size  # should be around 5.17

library(smotefamily)


train_data$Diabetes_binary <- as.numeric(as.factor(train_data$Diabetes_binary)) - 1


X <- train_data[, -which(names(train_data) == "Diabetes_binary")]
y <- train_data$Diabetes_binary

# Apply SMOTE with calculated dup_size
smote_result <- SMOTE(X, y, K = 5, dup_size = dup_size)

# Combine back into a data frame
train_balanced <- smote_result$data
colnames(train_balanced)[ncol(train_balanced)] <- "Diabetes_binary"
train_balanced$Diabetes_binary <- as.factor(train_balanced$Diabetes_binary)

table(train_balanced$Diabetes_binary)
prop.table(table(train_balanced$Diabetes_binary)) * 100


# Create summary for before and after
before <- prop.table(table(train_data$Diabetes_binary))
after  <- prop.table(table(train_balanced$Diabetes_binary))

# Create separate data frames
before_df <- data.frame(
  Class = c("Non-Diabetic", "Diabetic"),
  Proportion = as.numeric(before)
)

after_df <- data.frame(
  Class = c("Non-Diabetic", "Diabetic"),
  Proportion = as.numeric(after)
)

library(ggplot2)

ggplot(before_df, aes(x = Proportion, y = Class, fill = Class)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c("skyblue", "pink")) +
  labs(title = "Diabetes Distribution Before SMOTE",
       x = "Proportion", y = NULL) +
  theme_minimal() +
  theme(legend.position = "none",
        plot.title = element_text(size = 14, face = "bold", hjust = 0.5))

ggplot(after_df, aes(x = Proportion, y = Class, fill = Class)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c("skyblue", "pink")) +
  labs(title = "Diabetes Distribution After SMOTE",
       x = "Proportion", y = NULL) +
  theme_minimal() +
  theme(legend.position = "none",
        plot.title = element_text(size = 14, face = "bold", hjust = 0.5))

### Histogram

# Load required library
library(tidyverse)

# Create combined histogram using raw values (before scaling)
train_data %>%
  select(BMI, MentHlth, Age) %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value") %>%
  ggplot(aes(x = Value, fill = Variable)) +
  geom_histogram(bins = 30, color = "black", alpha = 0.7) +
  facet_wrap(~ Variable, scales = "free", ncol = 1) +
  labs(title = "Histogram of BMI, Mental Health Days, and Age (Before Feature Scaling)",
       x = "Original Value",
       y = "Frequency") +
  theme_minimal() +
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5, face = "bold"))


install.packages("patchwork")
library(patchwork)
library(tidyverse)

# Before scaling plot
p1 <- train_balanced %>%
  select(BMI, MentHlth, Age) %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value") %>%
  ggplot(aes(x = Value, fill = Variable)) +
  geom_histogram(bins = 30, color = "black", alpha = 0.6) +
  facet_wrap(~ Variable, scales = "free", ncol = 1) +
  labs(title = "Before Feature Scaling", x = "Standardized Value", y = "Frequency") +
  theme_minimal() +
  theme(legend.position = "none", plot.title = element_text(hjust = 0.5))
print(p1)

sapply(train_balanced[c("BMI", "MentHlth", "Age")], mean)
sapply(train_balanced[c("BMI", "MentHlth", "Age")], sd)


#####Feature Scaling (Standardization)

# Apply standardization to numeric variables
num_vars <- c("BMI", "MentHlth", "Age")
train_balanced[num_vars] <- scale(train_balanced[num_vars])
test_data[num_vars] <- scale(test_data[num_vars])

summary(train_balanced$BMI)
summary(train_balanced[num_vars])


# After scaling plot
p2 <- train_balanced %>%
  select(BMI, MentHlth, Age) %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value") %>%
  ggplot(aes(x = Value, fill = Variable)) +
  geom_histogram(bins = 30, color = "black", alpha = 0.6) +
  facet_wrap(~ Variable, scales = "free", ncol = 1) +
  labs(title = "After Feature Scaling", x = "Standardized Value", y = "Frequency") +
  theme_minimal() +
  theme(legend.position = "none", plot.title = element_text(hjust = 0.5))
print(p2)

# Combine plots
p1 / p2
(p1 / p2) + plot_annotation(title = "Comparison of Feature Distributions: Before vs After Scaling")


sapply(train_balanced[c("BMI", "MentHlth", "Age")], mean)

sapply(train_balanced[c("BMI", "MentHlth", "Age")], sd)


###################  Model Building  ################

#### Logistic Regression 

# Load necessary libraries
install.packages("caret")
library(caret)


train_balanced$Diabetes_binary <- as.factor(train_balanced$Diabetes_binary)
test_data$Diabetes_binary <- as.factor(test_data$Diabetes_binary)

str(train_balanced)

logit_model <- glm(Diabetes_binary ~ ., data = train_balanced, family = binomial)

summary(logit_model)

logit_pred_probs <- predict(logit_model, newdata = test_data, type = "response")
logit_pred_class <- ifelse(logit_pred_probs >= 0.5, 1, 0)
logit_pred_class <- factor(logit_pred_class, levels = levels(test_data$Diabetes_binary))

conf_matrix <- confusionMatrix(logit_pred_class, test_data$Diabetes_binary, positive = "1")
print(conf_matrix)


#####Feature Importance based on Logistic Regression

# logit_model <- glm(Diabetes_binary ~ ., data = train_balanced, family = binomial)

# Extract coefficients (excluding intercept)
coefficients <- coef(logit_model)[-1]  # Remove intercept

# Take absolute values to reflect importance
feature_importance <- abs(coefficients)

# Sort features by importance (increasing), then reverse for top-down
sorted_importance <- sort(feature_importance, decreasing = TRUE)

# Plot with most important feature on top
barplot(rev(sorted_importance),
        horiz = TRUE,
        las = 1,
        col = "steelblue2",
        main = "Feature Importance (Logistic Regression)",
        xlab = "Absolute Coefficient Value")


####Random forest

# Load necessary libraries
install.packages("randomForest")
library(randomForest)
library(caret)


train_balanced$Diabetes_binary <- as.factor(train_balanced$Diabetes_binary)
test_data$Diabetes_binary <- as.factor(test_data$Diabetes_binary)


set.seed(123)


rf_model <- randomForest(Diabetes_binary ~ HighBP + HighChol + BMI + PhysActivity +
                          GenHlth + MentHlth + DiffWalk + Age,
                        data = train_balanced,
                        ntree = 100,
                        mtry = 2,
                        importance = TRUE)


print(rf_model)

rfPredictions <- predict(rf_model, test_data)

confusionMatrix(rfPredictions, test_data$Diabetes_binary)
importance(rf_model)

# Plot variable importance
varImpPlot(rf_model)
varImpPlot(rf_model, pch = 20, col = "blue",
           main = "Random Forest Model - Diabetes Prediction")

# Predict probabilities (for AUC)
rf_pred_probs <- predict(rf_model, newdata = test_data, type = "prob")[, 2]  # Prob for class "1"


####Decision Tree
# Load necessary libraries
install.packages("rpart")
install.packages("rpart.plot")
install.packages("caret")

library(rpart)
library(rpart.plot)
library(caret)


train_balanced$Diabetes_binary <- as.factor(train_balanced$Diabetes_binary)
test_data$Diabetes_binary <- as.factor(test_data$Diabetes_binary)


dt_model <- rpart(Diabetes_binary ~ ., data = train_balanced, method = "class")
dt_pred <- predict(dt_model, newdata = test_data, type = "class")

dt_conf_matrix <- confusionMatrix(dt_pred, test_data$Diabetes_binary)
print(dt_conf_matrix)


#### Matrix Comparison

# Load required libraries
install.packages("pROC")
install.packages("caret")
install.packages("randomForest")

library(pROC)
library(caret)
library(randomForest)


train_balanced$Diabetes_binary <- as.factor(train_balanced$Diabetes_binary)
test_data$Diabetes_binary <- as.factor(test_data$Diabetes_binary)

set.seed(123)
rf_model <- randomForest(Diabetes_binary ~ ., 
                         data = train_balanced, 
                         ntree = 100,
                         importance = TRUE)


rf_pred_class <- predict(rf_model, newdata = test_data)
rf_pred_probs <- predict(rf_model, newdata = test_data, type = "prob")[, 2]

dt_probs <- predict(dt_model, newdata = test_data, type = "prob")[, 2]

calculate_metrics <- function(predicted, actual, probs = NULL, positive_class = "1") {
  cm <- confusionMatrix(predicted, actual, positive = positive_class)
  
  precision <- cm$byClass["Precision"]
  recall <- cm$byClass["Recall"]
  specificity <- cm$byClass["Specificity"]
  f1 <- cm$byClass["F1"]
  accuracy <- cm$overall["Accuracy"]
  auc <- if (!is.null(probs)) {
    roc_obj <- roc(actual, as.numeric(probs))
    as.numeric(auc(roc_obj))
  } else {
    NA
  }
  
  return(c(
    Accuracy = accuracy * 100,
    Precision = precision * 100,
    Recall = recall * 100,
    Specificity = specificity * 100,
    F1_Score = f1 * 100,
    AUC = auc * 100
  ))
}

# Calculate metrics for all models
logit_metrics <- calculate_metrics(logit_pred_class, test_data$Diabetes_binary, logit_pred_probs)
rf_metrics    <- calculate_metrics(rf_pred_class, test_data$Diabetes_binary, rf_pred_probs)
dt_metrics    <- calculate_metrics(dt_pred, test_data$Diabetes_binary, dt_probs)


# Combine into a result table
results_df <- rbind(
  Logistic_Regression = logit_metrics,
  Random_Forest = rf_metrics,
  Decision_Tree = dt_metrics
)


print(round(results_df, 2))


#### ROC Curve  
# Load required libraries
install.packages("pROC")
install.packages("ggplot2")
library(pROC)
library(ggplot2)

# Compute ROC and AUC for each model
roc_logit <- roc(test_data$Diabetes_binary, as.numeric(logit_pred_probs))
roc_rf    <- roc(test_data$Diabetes_binary, as.numeric(rf_pred_probs))
roc_dt    <- roc(test_data$Diabetes_binary, as.numeric(dt_probs))

# Extract AUCs and round
auc_logit <- round(auc(roc_logit), 3)
auc_rf    <- round(auc(roc_rf), 3)
auc_dt    <- round(auc(roc_dt), 3)

# Create label names with AUCs
model_labels <- c(
  paste0("Logistic Regression (AUC = ", auc_logit, ")"),
  paste0("Random Forest (AUC = ", auc_rf, ")"),
  paste0("Decision Tree (AUC = ", auc_dt, ")")
)

# Create a combined data frame
roc_data <- data.frame(
  fpr = c(1 - roc_logit$specificities, 1 - roc_rf$specificities, 1 - roc_dt$specificities),
  tpr = c(roc_logit$sensitivities, roc_rf$sensitivities, roc_dt$sensitivities),
  model = factor(c(rep(model_labels[1], length(roc_logit$sensitivities)),
                   rep(model_labels[2], length(roc_rf$sensitivities)),
                   rep(model_labels[3], length(roc_dt$sensitivities))))
)

# Plot ROC curves
ggplot(roc_data, aes(x = fpr, y = tpr, color = model)) +
  geom_line(size = 1.2) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "yellow") +
  labs(title = "ROC Curve Analysis",
       x = "False Positive Rate",
       y = "True Positive Rate",
       color = "Model (with AUC)") +
  theme_minimal() +
  theme(text = element_text(size = 14),
        legend.title = element_text(face = "bold"))




#####Prediction Testing on Simulated Patient Input

# Save the model to disk
saveRDS(logit_model, file = "logit_model.rds")
loaded_logit_model <- readRDS("logit_model.rds")

pred_probs <- predict(loaded_logit_model, newdata = test_data, type = "response")
pred_class <- ifelse(pred_probs >= 0.5, 1, 0)
pred_class <- factor(pred_class, levels = levels(test_data$Diabetes_binary))

conf_matrix <- confusionMatrix(pred_class, test_data$Diabetes_binary)
print(conf_matrix)


names(logit_model$model)
head(data, 5)
tail(data, 5)

create_patient <- function(HighBP, HighChol, BMI, PhysActivity, GenHlth, MentHlth, DiffWalk, Age) {
  data.frame(
    HighBP = as.integer(HighBP),
    HighChol = as.integer(HighChol),
    BMI = BMI,
    PhysActivity = as.integer(PhysActivity),
    GenHlth = GenHlth,
    MentHlth = MentHlth,
    DiffWalk = as.integer(DiffWalk),
    Age = Age
  )
}

# Create patient input
new_patient <- create_patient(1, 1, 18, 0, 4, 0, 1, 11) #253677th Patient

# Predict
prob <- predict(logit_model, newdata = new_patient, type = "response")
predicted_class <- ifelse(prob >= 0.5, 1, 0)

# Output
if (predicted_class == 1) {
  cat("Prediction: 1 (Diabetic)\n")
} else {
  cat("Prediction: 0 (Non-Diabetic)\n")
}


############ Cover Page(Word_Cloud) ############

install.packages("pdftools")
install.packages("tm")
install.packages("wordcloud")
install.packages("RColorBrewer")

library(pdftools)
library(tm)
library(wordcloud)
library(RColorBrewer)


text <- pdf_text("D:/R Class/Data analytics & intelligence_Individual Assignment.pdf")

full_text <- paste(text, collapse = " ")

corpus <- Corpus(VectorSource(full_text))

corpus <- tm_map(corpus, content_transformer(tolower))       
corpus <- tm_map(corpus, removePunctuation)                  
corpus <- tm_map(corpus, removeNumbers)                      
corpus <- tm_map(corpus, removeWords, stopwords("english"))  
corpus <- tm_map(corpus, stripWhitespace)                   

dtm <- TermDocumentMatrix(corpus)
matrix <- as.matrix(dtm)
word_freq <- sort(rowSums(matrix), decreasing = TRUE)
df <- data.frame(word = names(word_freq), freq = word_freq)

wordcloud(words = df$word,
          freq = df$freq,
          min.freq = 10,
          random.order = FALSE,
          colors = brewer.pal(8, "Dark2"))



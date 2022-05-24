# load libraries
library(caret)
library(randomForest)
library(superml)
library(scales)
library(dplyr)
library(moments)
library(psych)
library(e1071)
library(rpart)
library(rpart.plot)
library(forecast)
library(varImp)
library(pROC)
library(performance)

# load dataset
data <- read.csv("BankChurners.csv")

#******************************************************************************#
###Data Preprocessing

# Dropping columns
data <- subset(data, select = -c(CLIENTNUM, Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1,
          Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2))

# Checking for missing values
colSums(sapply(data, is.na))

#*****************************************************************************#

#*****************************************************************************#

###Data Exploration

# variable names
var_names = names(data)
var_names

# To check numeric varibles
numeric_var = select_if(data, is.numeric)
colSums(sapply(numeric_var, is.na))

# numeric summary
summary(numeric_var)

# To check categorical variables
categorical_var = select_if(data, is.factor)


# Exploring target feature
table(data$Attrition_Flag)

# Attrition flag on Education level
Education_Flag = with(data, table(Attrition_Flag, Education_Level))
Education_Flag

# Attrition flag on Income category
Income_Flag = with(data, table(Attrition_Flag, Income_Category))
Income_Flag

# Attrition flag on Card category
Card_Flag = with(data, table(Attrition_Flag, Card_Category))
Card_Flag

#*****************************************************************************#

#*****************************************************************************#

###Plottings

# Attrition Flag
p <- table(data$Attrition_Flag)
p
library(plotrix)
slices <- c(1627, 8500)
lbls <- paste(c("Attrited Customer", "Existing Customer")," ",pct, "%", sep = " ")
pie3D(slices,labels=lbls ,explode=0.35,main="Customer Attrition percentage")

#Transaction Count

ggplot(data = data) +
  geom_point(mapping = aes(x = Total_Trans_Ct, y = Total_Trans_Amt, color = Dependent_count)) +
  labs(x = "Total Transaction Count", y = "Total Transaction Amount") + labs(colour = "Dependent Count")

#Customer Age, Income Category and Genders

ggplot(data = data) +
  geom_point(mapping = aes(x = Customer_Age, y = Income_Category, color = Gender)) + 
  facet_wrap(~ Card_Category, nrow = 2) +
  labs(x = "Customer Age", y = "Income Category")

#Customer Age, Months on book and Marital Status

ggplot(data = data) +
  geom_point(mapping = aes(x = Customer_Age, y = Months_on_book, color = Marital_Status)) +
  geom_smooth(mapping = aes(x = Customer_Age, y = Months_on_book)) +
  labs(x = "Customer Age", y = "Months on Book") + labs(color = " Marital Status")


# Months inactive in one year, Total number of transactions & Attrition Flag

ggplot(data,aes(x = Months_Inactive_12_mon, y = Total_Trans_Ct, fill = Attrition_Flag))+
  geom_bar(stat= "identity", position = "dodge") + 
  labs(x = "Months Inactive in one year", y = "Total number of transactions") + labs(fill = "Attrition Flag")

#Total Transaction Count in Q4 over Q1

ggplot(data = data, mapping = aes(x = Total_Ct_Chng_Q4_Q1, colour = Attrition_Flag)) +
  geom_freqpoly(binwidth = 0.1) + 
  labs(x = "Total Transaction Count in Q4 over Q1") + labs(color = "Attrition Flag")

#Total Amount change in Q4 over Q1

ggplot(data = card, mapping = aes(x = Total_Amt_Chng_Q4_Q1, colour = Attrition_Flag)) +
  geom_freqpoly(binwidth = 0.1) + 
  labs(x = "Total Amount Change in Q4 over Q1") + labs(color = "Attrition Flag")

#Number of Contacts in one year

ggplot(data = data) + 
  geom_bar(mapping = aes(x = Contacts_Count_12_mon, fill = Attrition_Flag), position = "fill", color = "green") +
  labs(x = "Number of contacts in one year") + labs(fill = "Attrition Flag")


#*****************************************************************************#

#*****************************************************************************#

###Feature Engineering

## Encoders

# Using label encoder
label <- LabelEncoder$new()

data$Attrition_Flag <- label$fit_transform(data$Attrition_Flag)
data$Gender <- label$fit_transform(data$Gender)
data$Education_Level <- label$fit_transform(data$Education_Level)
data$Marital_Status <- label$fit_transform(data$Marital_Status)
data$Income_Category <- label$fit_transform(data$Income_Category)
data$Card_Category <- label$fit_transform(data$Card_Category)

# Using Onehot encoding
#dummy <- dummyVars(" ~ .", data = data)
#newdata <- data.frame(predict(dummy, newdata = data))

# Calculate Skewness
skewness <- skewness(data$Attrition_Flag)
plot(skewness)

# log transformation
log_data <- log(data$Attrition_Flag)
hist(log_data, col = 'red')

# Square root transformation
sqrt_data <- sqrt(data$Attrition_Flag)
hist(sqrt_data, col = 'blue')

# Cube root transformation
cbrt_data <- (data$Attrition_Flag)^(1/3)
hist(cbrt_data, col = 'orange')

## Scaling
# Feature Scaling
data_scaled <- as.data.frame(scale(data))
data_scaled

# MinMaxScaler
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

data$Attrition_Flag <- normalize(data$Attrition_Flag)
data$Customer_Age <- normalize(data$Customer_Age)
data$Gender <- normalize(data$Dependent_count)
data$Education_Level <- normalize(data$Education_Level)
data$Marital_Status <- normalize(data$Marital_Status)
data$Income_Category <- normalize(data$Income_Category)
data$Card_Category <- normalize(data$Card_Category)
data$Months_on_book <- normalize(data$Months_on_book)
data$Total_Relationship_Count <- normalize(data$Total_Relationship_Count)
data$Months_Inactive_12_mon <- normalize(data$Months_Inactive_12_mon)
data$Contacts_Count_12_mon <- normalize(data$Contacts_Count_12_mon)
data$Credit_Limit <- normalize(data$Credit_Limit)
data$Total_Revolving_Bal <- normalize(data$Total_Revolving_Bal)
data$Avg_Open_To_Buy <- normalize(data$Avg_Open_To_Buy)
data$Total_Amt_Chng_Q4_Q1 <- normalize(data$Total_Amt_Chng_Q4_Q1)
data$Total_Trans_Amt <- normalize(data$Total_Trans_Amt)
data$Total_Trans_Ct <- normalize(data$Total_Trans_Ct)
data$Total_Ct_Chng_Q4_Q1 <- normalize(data$Total_Ct_Chng_Q4_Q1)
data$Avg_Utilization_Ratio <- normalize(data$Avg_Utilization_Ratio)

#*****************************************************************************#

#*****************************************************************************#

### Data Preparation for modelling

# splitting dataset into training & testing sets
training_index <- createDataPartition(data$Attrition_Flag,p=0.7,list=FALSE)
trainingset <- data[training_index,]
testingset <- data[-training_index,]


#*****************************************************************************#

#*****************************************************************************#

### PCA - Principal Component analysis

train_x <- trainingset[,2:20]
train_y <- trainingset[,1]

pca_train <- prcomp(train_x,scale. = T)
attributes(pca_train)

loadings <- as.data.frame(pca_train$x)

pca_train$scale

print(pca_train)

matrix <- pca_train$rotation

# variance of each pc
std_dev <- pca_train$sdev
pr_comp_var <- std_dev^2
pr_comp_var

# ratio of variance by each component
prop_var_ex <- pr_comp_var/sum(pr_comp_var)
prop_var_ex

# pca chart
plot(cumsum(prop_var_ex), xlab = "Principal Component",
                ylab = "Proportion of Variance Explained",type = "b")

# concatenate dependent variable
pca_train2 <- cbind(loadings,train_y)



loadings2 <- loadings[1:8]
pca_train2 <- cbind(loadings2,train_y)


#Linear model
lin_model <- lm(train_y ~.,data=pca_train2)
summary(lin_model)

# pca on testing set
pca_test <- testingset[,2:20]
pca_test2 <- predict(pca, newdata = pca_test)

pca_test2 <- as.data.frame(pca_test2)
pca_test3 <- pca_test2[1:8]
test_y <- testingset$Attrition_Flag

# prediction
predict_pca <- predict(lin_model, pca_test2)

# Calculate r square
error <- test_y - predict_pca
mse <- mean(error^2)
R2=1-sum(error^2)/sum((test_y- mean(test_y))^2)
R2

#*****************************************************************************#

#*****************************************************************************#

### Model Selection

dim(trainingset)
dim(testingset)

dim(data)

#***********************linear model********************************##

linear <- lm(Attrition_Flag ~ ., data = trainingset)
summary(linear)

#Predict Output
lin_predicted <- predict(linear,testingset)

#**********************logistic model******************************##

logistic <- glm(Attrition_Flag ~ ., data = trainingset,family='binomial')
summary(logistic)

#Predict Output
log_predicted <- predict(logistic,testingset)
#*****************************************************************##

#**********************Decision tree*****************************##

tree <- rpart(Attrition_Flag ~ ., data = trainingset ,method="class",
             control = rpart.control(cp = 0.01))

summary(tree)

#Predict output 
tree_predicted <- predict(tree,testingset, type='prob')

# plot cp
plotcp(tree)
printcp(tree)

# plot tree
rpart.plot(tree, box.palette="auto",
           branch.lty=3, shadow.col=0, nn=TRUE, tweak=1, extra = TRUE)

# pruned classification tree
ptree <- prune(tree,cp=tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"])

rpart.plot(ptree, uniform=TRUE,main="Pruned Classification Tree")

# Accuracy
tree.preds <- predict(tree, testingset, type="prob")[, 2]
cm_tree <- table(testingset$Attrition_Flag,tree.preds)
accuracy <- (sum(diag(cm_tree))/sum(cm_tree)) 
accuracy

# ROC Curve
tree.roc <- roc(testingset$Attrition_Flag, tree.preds)
print(tree.roc)
plot(tree.roc, print.auc=TRUE, main="Decision Tree ROC")


#******************************************************************#

#****************************SVM***********************************#

svm <- svm(Attrition_Flag ~ ., data = trainingset)
summary(svm)
plot(svm, trainingset)

p1 <- predict(svm, testingset)
confusionMatrix(p1, testingset$Attrition_Flag)


#Predict Output 
svm_predicted= predict(svm,testingset)
plot(svm_predicted)

# ROC Curve
svm.roc <- roc(testingset$Attrition_Flag, svm_predicted)
print(svm.roc)
plot(svm.roc, print.auc=TRUE, main="SVM ROC")

#******************************************************************#

#****************************Naivebayes***************************#

naive <-naiveBayes(Attrition_Flag ~ ., data = trainingset)
summary(naive)

#Predict Output 
naive_predicted= predict(naive,testingset)

#******************************************************************#

#***************************RandomForest**************************#

forest <- randomForest(Attrition_Flag ~ ., trainingset,importance =TRUE,
                    ntree=500,nodesize=7, na.action=na.roughfix)
summary(forest)

# plotting the graph
varImpPlot(forest, type=1)

#Predict Output
forest_predicted= predict(forest,testingset)
accuracy(forest_predicted, trainingset$Attrition_Flag)

plot(forest)

# ROC Curve
forest.roc <- roc(testingset$Attrition_Flag, forest_predicted)
print(forest.roc)
plot(forest.roc, print.auc=TRUE, main="RandomForest ROC")


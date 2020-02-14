library(readxl)
library(dplyr)
library(caTools)
library(mltools)
library(MASS)
library(leaps)
library(glmnet)
library(ISLR)
library(car)
setwd("C:/Users/hungu/Documents/Black-Friday-Sales")

train = read.csv('train.csv')
test = read.csv('test.csv')

# Preprocessing

Purchase = 0
test = cbind(test, Purchase)

data = rbind(train, test)

str(data)

colSums(is.na(data))

summary(data$Gender)
summary(data$Age)
summary(data$Stay_In_Current_City_Years)

levels(data$Stay_In_Current_City_Years)
data$Stay_In_Current_City_Years = as.factor(as.numeric(data$Stay_In_Current_City_Years) - 1)

data$Marital_Status = as.factor(data$Marital_Status)
summary(data$Marital_Status)

data$Occupation = as.factor(data$Occupation)
summary(data$Occupation)

summary(as.factor(data$Product_Category_2))
data = data %>% mutate(Product_Category_2=replace(Product_Category_2, is.na(Product_Category_2), -1))
data$Product_Category_2 = as.factor(data$Product_Category_2)
summary(data$Product_Category_2)

summary(as.factor(data$Product_Category_3))
data = data %>% mutate(Product_Category_3=replace(Product_Category_3, is.na(Product_Category_3), -1))
data$Product_Category_3 = as.factor(data$Product_Category_3)
summary(data$Product_Category_3)

data$Product_Category_1 = as.factor(data$Product_Category_1)
summary(data$Product_Category_1)

data = data[-c(1, 2)]

str(data)

# Segregating train test data

train_ = data[1:nrow(train),]
test_ = data[(nrow(train) + 1):nrow(data),]

# Splitting train data into train-validation set

set.seed(123)
split = sample.split(train_$Purchase, SplitRatio = 0.8)
training_set = subset(train_, split==TRUE)
validation_set = subset(train_, split==FALSE)

X_train = model.matrix(Purchase ~ ., data = training_set)[,-1]
X_val = model.matrix(Purchase ~ ., data = validation_set)[,-1]

# Normal Linear Regression

lm.fit = lm(Purchase ~ ., data = training_set)
summary(lm.fit)
pred_lm = predict(lm.fit, newx = X_val)
sqrt(mean((validation_set$Purchase - pred_lm)^2))

# Subsetting

nvmax_ = 9

subset.fit = regsubsets(Purchase ~ ., data = training_set, method='backward', nvmax = nvmax_)
k = summary(subset.fit)
k$adjr2

val.errors = rep(NA, nvmax_)
for(i in 1:nvmax_){
  coefi = coef(subset.fit, id=i)
  pred = cbind(1, X_val[,names(coefi[-c(1)])]) %*% coefi
  val.errors[i]= sqrt(mean((validation_set$Purchase - pred) ^2))
}
val.errors
val.errors[which.min(val.errors)]

# Ridge and Lasso

ld = 10^seq(10, -2, length=100)

y = training_set$Purchase
rig.fit = glmnet(X_train, y, alpha=1, lambda=ld)
plot(rig.fit)

cvglm = cv.glmnet(X_train, y, alpha=1, lambda=ld, nfolds=5)
plot(cvglm)

best = cvglm$lambda.min
best

predict(rig.fit, s=best, type='coefficients')

pred_rl = predict(rig.fit, s=best, newx = X_val)
sqrt(mean((validation_set$Purchase - pred_rl)^2))

# Predicting Test Set

test_set = model.matrix(Purchase ~ ., data = test_)[,-1]
pred_test = predict(rig.fit, s=best, newx = test_set)
pred_test[pred_test < 0]

submission = data.frame('Purchase'=pred_test,
                        'User_ID'=test$User_ID,
                        'Product_ID'=test$Product_ID)
names(submission)[1] = 'Purchase'

write.csv(submission, 'RL.csv', row.names = F)
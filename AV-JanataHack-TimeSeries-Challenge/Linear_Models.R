library(dplyr)
library(caTools)
library(mltools)
library(MASS)
library(leaps)
library(glmnet)
library(ISLR)
library(tidyverse)
library(car)

setwd("C:\\Users\\hungu\\Documents\\AV-JanataHack-TimeSeries")

train = read.csv('train.csv')
test = read.csv('test.csv')

electricity_consumption = 0
test = cbind(test, electricity_consumption)

data = rbind(train, test)

colSums(is.na(data))
data = separate(data, "datetime", c("Date", "Time"), sep = " ")
data = separate(data, "Date", c("Year", "Month", "Day"), sep = "-")
data = separate(data, "Time", c("Hour"), sep = ":")

data$Year = as.factor(data$Year)
data$Month = as.factor(data$Month)
data$Day = as.factor(data$Day)
data$Hour = as.factor(data$Hour)
data$var2 = as.factor(data$var2)

train_ = data[1:nrow(train),-c(1)]
test_ = data[(nrow(train) + 1):nrow(data),-c(1)]

set.seed(123)
split = sample.split(train_$electricity_consumption, SplitRatio = 0.8)
training_set = subset(train_, split==TRUE)
testing_set = subset(train_, split==FALSE)

X = model.matrix(electricity_consumption ~ ., data = training_set)[,-1]
X_test = model.matrix(electricity_consumption ~ ., data = testing_set)[,-1]

# Linear Regression

lm.fit = lm(electricity_consumption ~ ., data = training_set)
summary(lm.fit)
pred_lm = predict(lm.fit, newx = X_test)
sqrt(mean((testing_set$electricity_consumption - pred_lm)^2))

# Subsetting

nvmax_ = 9

subset.fit = regsubsets(electricity_consumption ~ ., data = training_set, method='backward', nvmax = nvmax_)
k = summary(subset.fit)
k$adjr2

val.errors = rep(NA, nvmax_)
for(i in 1:nvmax_){
  coefi = coef(subset.fit, id=i)
  pred = cbind(1, X_test[,names(coefi[-c(1)])]) %*% coefi
  val.errors[i]= sqrt(mean((testing_set$electricity_consumption - pred) ^2))
}
val.errors
val.errors[which.min(val.errors)]


# Ridge and Lasso

ld = 10^seq(10, -2, length=100)

y = training_set$electricity_consumption
rig.fit = glmnet(X, y, alpha=1, lambda=ld)
plot(rig.fit)
cvglm = cv.glmnet(X, y, alpha=1, lambda=ld, nfolds=5)
plot(cvglm)
best = cvglm$lambda.min
best
predict(rig.fit, s=best, type='coefficients')

pred_rl = predict(rig.fit, s=best, newx = X_test)
sqrt(mean((testing_set$electricity_consumption - pred_rl)^2))

# Predicting the test set
test_set = model.matrix(electricity_consumption ~ ., data = test_)[,-1]
pred_test = predict(rig.fit, s=best, newx = test_set)

submission = data.frame('ID'=test$ID, pred_test)
names(submission)[2] = 'electricity_consumption'

write.csv(submission, 'Ridge_Lasso.csv', row.names = F)
write.csv(train_, 'CleanTrain.csv', row.names = F)
write.csv(test_, 'CleanTest.csv', row.names = F)
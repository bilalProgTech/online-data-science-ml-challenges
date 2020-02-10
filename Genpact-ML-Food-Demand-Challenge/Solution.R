library(readxl)
library(dplyr)
library(caTools)
library(mltools)
library(MASS)
library(leaps)
library(glmnet)
library(ISLR)
library(car)

train = read.csv("train.csv")
test = read.csv('test.csv')
meal = read.csv('meal_info.csv')
fulfillment = read.csv('fulfilment_center_info.csv')

# Preprocessing

num_orders = 0
test = cbind(test, num_orders)

data = rbind(train, test)
data = merge(data, meal, by = 'meal_id')
data = merge(data, fulfillment, by='center_id')

data = data[-c(1, 2)]

data = data %>% arrange(week)

sum(is.na(data))

str(data)

summary(data$category)
summary(data$cuisine)
summary(data$center_type)

boxplot(log1p(data$num_orders))
data$num_orders = log1p(data$num_orders)

boxplot(data$checkout_price)
boxplot(data$base_price)

data$checkout_price = scale(data$checkout_price)
data$base_price = scale(data$base_price)

data$emailer_for_promotion = as.factor(data$emailer_for_promotion)
data$homepage_featured = as.factor(data$homepage_featured)

boxplot(data$op_area)

str(data)

train_ = data[1:nrow(train),]
test_ = data[(nrow(train) + 1):nrow(data),]

train_set = train_[train$week <= 140,]
val_set = train_[train_$week > 140, ]

X = model.matrix(num_orders ~ .-week-id, data = train_set)[,-1]
X_val = model.matrix(num_orders ~ .-week-id, data = val_set)[,-1]

# Normal Linear Regression

lm.fit = lm(num_orders ~ .-week-id, data = train_set)
summary(lm.fit)
pred_lm = predict(lm.fit, newx = X_val)
sqrt(mean((val_set$num_orders - pred_lm)^2))

# Subsetting

nvmax_ = 10

subset.fit = regsubsets(num_orders ~ .-week-id, data = train_set, method='backward', nvmax = nvmax_)
k = summary(subset.fit)
k$adjr2

val.errors = rep(NA, nvmax_)
for(i in 1:nvmax_){
  coefi = coef(subset.fit, id=i)
  pred = cbind(1, X_val[,names(coefi[-c(1)])]) %*% coefi
  val.errors[i]= sqrt(mean((exp(val_set$num_orders) - exp(pred)) ^2))
}
val.errors
val.errors[which.min(val.errors)]

# Ridge and Lasso

ld = 10^seq(10, -2, length=150)

y = train_set$num_orders
rig.fit = glmnet(X, y, alpha=1, lambda=ld)
plot(rig.fit)
cvglm = cv.glmnet(X, y, alpha=1, lambda=ld, nfolds=10)
plot(cvglm)
best = cvglm$lambda.min
best
predict(rig.fit, s=best, type='coefficients')

pred_rl = predict(rig.fit, s=best, newx = X_val)
sqrt(mean((val_set$num_orders - pred_rl)^2))

# Predicting Test Set

test_set = model.matrix(num_orders ~ .-week-id, data = test_)[,-1]
pred_test = predict(rig.fit, s=best, newx = test_set)
pred_test[pred_test < 0]

submission = data.frame('id'=test$id,
                        'num_orders'=exp(pred_test))
names(submission)[2] = 'num_orders'

write.csv(submission, 'RL.csv', row.names = F)
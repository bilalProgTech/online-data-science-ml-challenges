library(readxl)
library(dplyr)
library(caTools)
library(mltools)
library(MASS)
library(leaps)
library(glmnet)
library(ISLR)
library(car)

train = read.csv("C:/Users/hungu/Documents/Big-Mart/train.csv")
test = read.csv("C:/Users/hungu/Documents/Big-Mart/test.csv")

Item_Outlet_Sales = 0
test = cbind(test, Item_Outlet_Sales)

data = rbind(train, test)

colSums(is.na(data))

data = data %>% mutate(Item_Weight=replace(Item_Weight, is.na(Item_Weight), round(mean(Item_Weight, na.rm = T),2)))

summary(as.factor(data$Outlet_Size))
data$Outlet_Size = ifelse(data$Outlet_Size=='Small','Small',ifelse(data$Outlet_Size=='High','High', 'Medium'))

sum(is.na(data))

data$Item_Fat_Content = ifelse(data$Item_Fat_Content=='LF' | data$Item_Fat_Content=='low fat','Low Fat',
                          ifelse(data$Item_Fat_Content=='reg' | data$Item_Fat_Content=='Regular','Regular', 
                                 'Low Fat'))

summary(as.factor(data$Item_Fat_Content))
summary(as.factor(data$Outlet_Size))
summary(as.factor(data$Item_Type))
summary(as.factor(data$Outlet_Identifier))
summary(as.factor(data$Outlet_Location_Type))
summary(as.factor(data$Outlet_Type))

data %>% group_by(Item_Fat_Content) %>% 
  summarise_at(vars(Item_Outlet_Sales), funs(mean))

boxplot(data$Item_Visibility)

Item_Outlet_Sales = data$Item_Outlet_Sales
data = as.data.frame(model.matrix(Item_Outlet_Sales ~ .-Item_Identifier, data = data)[,-1])
data = cbind(data, Item_Outlet_Sales)

train_ = data[1:nrow(train),]
test_ = data[(nrow(train) + 1):nrow(data),]

set.seed(123)
split = sample.split(train_$Item_Outlet_Sales, SplitRatio = 0.8)
training_set = subset(train_, split==TRUE)
testing_set = subset(train_, split==FALSE)

# Linear Regression
lm.fit = lm(Item_Outlet_Sales ~ ., data = training_set)
summary(lm.fit)
pred_lm = predict(lm.fit, newx = X_test)
sqrt(mean((testing_set$Item_Outlet_Sales - pred_lm)^2))

# Subsetting

nvmax_ = 10

subset.fit = regsubsets(Item_Outlet_Sales ~ ., data = training_set, method='backward', nvmax = nvmax_)
k = summary(subset.fit)
k$adjr2

X = model.matrix(Item_Outlet_Sales ~ ., data = training_set)[,-1]
X_test = model.matrix(Item_Outlet_Sales ~ ., data = testing_set)[,-1]

val.errors = rep(NA, nvmax_)
for(i in 1:nvmax_){
  coefi = coef(subset.fit, id=i)
  pred = cbind(1, X_test[,names(coefi[-c(1)])]) %*% coefi
  val.errors[i]= sqrt(mean((testing_set$Item_Outlet_Sales - pred) ^2))
}
val.errors
val.errors[which.min(val.errors)]

# Ridge and Lasso

ld = 10^seq(10, -2, length=100)

y = training_set$Item_Outlet_Sales
rig.fit = glmnet(X, y, alpha=1, lambda=ld)
plot(rig.fit)
cvglm = cv.glmnet(X, y, alpha=1, lambda=ld, nfolds=5)
plot(cvglm)
best = cvglm$lambda.min
best
predict(rig.fit, s=best, type='coefficients')

pred_rl = predict(rig.fit, s=best, newx = X_test)
mean(testing_set$Item_Outlet_Sales) - mean(pred_rl)
sqrt(mean((testing_set$Item_Outlet_Sales - pred_rl)^2))
mean(pred_rl)

# Predicting the test set
test_set = model.matrix(Item_Outlet_Sales ~ ., data = test_)[,-1]
pred_test = predict(rig.fit, s=best, newx = test_set)

submission = data.frame('Item_Identifier'=test$Item_Identifier,
                        'Outlet_Identifier'=test$Outlet_Identifier,pred_test)
names(submission)[3] = 'Item_Outlet_Sales'

write.csv(submission, 'C:/Users/hungu/Documents/Big-Mart/Ridge_Lasso.csv', row.names = F)
# train_vif$Item_Fat_Content = as.numeric(factor(train_vif$Item_Fat_Content))
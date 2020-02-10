library(leaps)
library(glmnet)
library(caret)
library(caTools)
library(boot)
library(MASS)
library(ISLR)
library(car)
library(mltools)
library(readxl)
library(dplyr)

train = read_excel("case_study_ridge_lasso.xls")
test = read_excel("case_study_ridge_lasso.xls", sheet = 2)

SalePrice = 0
test = cbind(test, SalePrice)
combi = rbind(train, test)

colSums(is.na(combi))
sum(is.na(combi))

str(combi)

combi = combi %>% mutate(LotFrontage=replace(LotFrontage, LotFrontage=="NA",NA))
summary(factor(combi$LotFrontage))
combi$LotFrontage = as.numeric(combi$LotFrontage)
combi = combi %>% 
  mutate(LotFrontage=replace(LotFrontage, 
                             is.na(LotFrontage),
                             round(mean(LotFrontage, na.rm = T),2)))

combi = combi %>% mutate(MasVnrArea=replace(MasVnrArea, MasVnrArea=="NA", NA))
summary(factor(combi$MasVnrArea))
combi$MasVnrArea = as.numeric(combi$MasVnrArea)
combi = combi %>% mutate(MasVnrArea=replace(MasVnrArea, is.na(MasVnrArea),round(mean(MasVnrArea, na.rm = T),2)))

combi = combi %>% mutate(GarageCars=replace(GarageCars, is.na(GarageCars),round(mean(GarageCars, na.rm = T),2)))
combi = combi %>% mutate(GarageArea=replace(GarageArea, is.na(GarageArea),round(mean(GarageArea, na.rm = T),2)))

combi$BsmtFinSF1 = as.numeric(combi$BsmtFinSF1)
combi = combi %>% mutate(BsmtFinSF1=replace(BsmtFinSF1, is.na(BsmtFinSF1),round(mean(BsmtFinSF1, na.rm = T),2)))

combi$BsmtFinSF2 = as.numeric(combi$BsmtFinSF2)
combi = combi %>% mutate(BsmtFinSF2=replace(BsmtFinSF2, is.na(BsmtFinSF2),round(mean(BsmtFinSF2, na.rm = T),2)))

combi$TotalBsmtSF = as.numeric(combi$TotalBsmtSF)
combi = combi %>% 
  mutate(TotalBsmtSF=replace(TotalBsmtSF, 
                             is.na(TotalBsmtSF),
                             round(mean(TotalBsmtSF, na.rm = T),2)))

combi$BsmtUnfSF = as.numeric(combi$BsmtUnfSF)
combi = combi %>% mutate(BsmtUnfSF=replace(BsmtUnfSF, is.na(BsmtUnfSF),round(mean(BsmtUnfSF, na.rm = T),2)))

summary(factor(combi$MasVnrType))
combi = combi %>% mutate(MasVnrType=replace(MasVnrType, MasVnrType=="NA","None"))
summary(factor(combi$MasVnrType))

summary(factor(combi$GarageYrBlt))
combi = combi %>% mutate(GarageYrBlt=replace(GarageYrBlt, GarageYrBlt=="NA",NA))
summary(factor(combi$GarageYrBlt))

index = which(is.na(combi$GarageYrBlt))
combi[index,'GarageYrBlt'] = combi[index,'YearBuilt']
combi$GarageYrBlt = as.numeric(combi$GarageYrBlt)

which(is.na(combi$GarageYrBlt))

combi$ExterQual = ordered(combi$ExterQual)
combi$ExterCond = ordered(combi$ExterCond)
combi$BsmtQual = ordered(combi$BsmtQual)
combi$BsmtCond = ordered(combi$BsmtCond)
combi$HeatingQC = ordered(combi$HeatingQC)
combi$KitchenQual = ordered(combi$KitchenQual)
combi$FireplaceQu = ordered(combi$FireplaceQu)
combi$GarageQual = ordered(combi$GarageQual)
combi$GarageCond = ordered(combi$GarageCond)
combi$PoolQC = ordered(combi$PoolQC)

combi = as.data.frame(unclass(combi))

str(combi)
sum(is.na(combi))

train_ = combi[1:nrow(train),]
test_ = combi[(nrow(train) + 1):nrow(combi),]

set.seed(123)
split = sample.split(train_$SalePrice, SplitRatio = 0.8)
training_set = subset(train_, split==TRUE)
testing_set = subset(train_, split==FALSE)

X = model.matrix(SalePrice ~.-Id, data=training_set)[,-1]
X_test = model.matrix(SalePrice ~.-Id, data=testing_set)[,-1]

#training_set = training_set[-c(1)]
#testing_set = testing_set[-c(1,81)]

ld = 10^seq(10, -2, length=100)

y = training_set$SalePrice
rig.fit = glmnet(X, y, alpha=1, lambda=ld)
plot(rig.fit)

cvglm = cv.glmnet(X, y, alpha=1, lambda=ld, nfolds=5)
plot(cvglm)

best = cvglm$lambda.min
best

predict(rig.fit, s=best, type='coefficients')

pred_rl = predict(rig.fit, s=best, newx = X_test)
sqrt(mean((log(testing_set$SalePrice) - log(pred_rl))^2))

row.names(test_) = NULL
test_set1 = model.matrix(SalePrice ~.-Id, data=test_)[,-1]
pred_test = predict(rig.fit, s=best, newx = test_set1)

df = as.data.frame(cbind(test_$Id, pred_test))

names(df)[1] = 'Id'
names(df)[2] = 'SalePrice'

write.csv(df, "C:/Users/hungu/Documents/MTech DS Docs/ASL/Week-code/sub1.csv", row.names = F)
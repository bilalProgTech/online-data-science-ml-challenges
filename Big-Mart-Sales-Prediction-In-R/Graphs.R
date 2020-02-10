setwd('C:/Users/hungu/Documents/MTech DS Docs/R Programming/Project')
#install.packages('corrplot')
#install.packages('cowplot')
#install.packages('mltools')
#install.packages('ISLR')
#install.packages('DMwR')
#install.packages('Metrics')

# To import data in datafram
library(data.table)
# To make used of pipelining process and cleaning of data and descriptive analysis of data
library(dplyr)
# To visualize the data
library(ggplot2)
# To make a correlation plot
library(corrplot)
# To tune the model, and for feature selection
library(caret)
# To clean the data
library(tidyverse)
# To create a correlation heatmap
library(cowplot)
# To determine metrics of model
library(Metrics)

# Loading the data
train = read.csv('Train.csv')

# Understanding the data
dim(train)
head(train)
summary(train)
glimpse(train)
names(train)

# Replacing Values
train <- train %>% mutate(Item_Fat_Content = replace(Item_Fat_Content,Item_Fat_Content == "LF","Low Fat"))
train <- train %>% mutate(Item_Fat_Content = replace(Item_Fat_Content,Item_Fat_Content == "low fat","Low Fat"))
train <- train %>% mutate(Item_Fat_Content = replace(Item_Fat_Content,Item_Fat_Content == "reg","Regular"))

#Replacing missing values in Outlet Size with "Small" as both have similar distribution with Outlet Sales
train$Outlet_Size[train$Outlet_Size==''] <- "Small"

# Missing Value Treatment
missing_values <-  summarise_all(train, funs(missing=sum(is.na(.))))

loc_na <-  which(is.na(train$Item_Weight))

for(i in loc_na){
  identifier <-  train$Item_Identifier[i]
  train$Item_Weight[i] <-  mean(train$Item_Weight[train$Item_Identifier==identifier], na.rm=T)
}

# To replace zero with mean
zero_index <- which(train$Item_Visibility==0)
for(i in zero_index){
  item <- train$Item_Identifier[i]
  train$Item_Visibility[i]=mean(train$Item_Visibility[train$Item_Identifier==item],na.rm=T)
}

# Continuous Data Analysis (Univariate) Creatin Histograms

h1 <-  ggplot(train,aes(Item_Visibility))+
  geom_histogram(bins = 100,binwidth=0.01,color='Black',fill='Sky Blue') +
  ylab('Count') +
  ggtitle("Item Visibility Count") +
  theme(plot.title = element_text(hjust = 0.5))

h2 <- ggplot(train,aes(Item_Weight)) +
  geom_histogram(bins = 100,color='Black',fill='Sky Blue') +
  ylab('Count') + 
  ggtitle("Item Weight Count") + 
  theme(plot.title = element_text(hjust = 0.5))

h3 <- ggplot(train,aes(Item_MRP)) + 
  geom_histogram(bins = 100,color='Black',fill='Sky Blue') + 
  ylab('Count') + 
  ggtitle("Item MRP Count") + 
  theme(plot.title = element_text(hjust = 0.5))

# Creating a canvas and displaying the plots
second_row_2 <-  plot_grid(h1, h2, ncol = 2)
plot_grid(h3, second_row_2, nrow = 2)

# Bivariate analysis of continuous variables

# Item_Weight vs Item_Outlet_Sales (Scatter Plot)
p1 <-  train %>% ggplot(aes(Item_Weight, Item_Outlet_Sales)) + 
  geom_point(color='tomato4') +
  ggtitle("Item Weight Vs Sales") +
  theme(plot.title = element_text(hjust = 0.5))

# Item_MRP vs Item_Outlet_Sales (Scatter Plot)
p2 <-  train %>% ggplot(aes(Item_MRP, Item_Outlet_Sales)) + 
  geom_point(color='tomato4') +
  ggtitle("Item MRP Vs Sales") +
  theme(plot.title = element_text(hjust = 0.5))

# Item_Visibility vs Item_Outlet_Sales (Scatter Plot)
p3 <-  train %>% ggplot(aes(Item_Visibility, Item_Outlet_Sales)) + 
  geom_point(color='tomato4') + 
  ggtitle("Item Visibility Vs Sales") + 
  theme(plot.title = element_text(hjust = 0.5))

second_row_2 <-  plot_grid(p1, p2, ncol = 2)
plot_grid(p3, second_row_2, nrow = 2)


# Outlier Detection using IQR Method in continuous variables

# Boxplot for each variable
boxplot(train$Item_Weight,xlab='Item_Weigth')
boxplot(train$Item_Visibility,xlab='Item_Visibility')
boxplot(train$Item_MRP,xlab='Item_MRP')

Q1_visibility <-  quantile(train$Item_Visibility)[2]
Q3_visibility <-  quantile(train$Item_Visibility)[4]
IQR_visibility <-  Q3_visibility - Q1_visibility

lower_visibility <-  Q1_visibility - 1.5 * IQR_visibility
upper_visibility <-  Q3_visibility + 1.5 * IQR_visibility

# Categorical Data Analysis

# Creating a data table for each categorical which determines Mean and Count of each unique values

# Item_Fat_Content
fat_bysales <-  train %>% group_by(Item_Fat_Content) %>% 
  summarise_at(vars(Item_Outlet_Sales), funs(Count = n(),Sales_Mean = mean))

# Item_Type
type_bysales <-  train %>% group_by(Item_Type) %>% 
  summarise_at(vars(Item_Outlet_Sales), funs(Count = n(),Sales_Mean = mean))

# Outlet_Size
outletsize_bysales = train %>% group_by(Outlet_Size) %>% 
  summarise_at(vars(Item_Outlet_Sales), funs(Count = n(),Sales_Mean = mean))

# Outlet_Location_Type
outletlocation_bysales = train %>% group_by(Outlet_Location_Type) %>% 
  summarise_at(vars(Item_Outlet_Sales), funs(Count = n(),Sales_Mean = mean))

# Outlet_Type
outlettype_bysales = train %>% group_by(Outlet_Type) %>% 
  summarise_at(vars(Item_Outlet_Sales), funs(Count = n(),Sales_Mean = mean))

# Outlet_Identifier
outletid_bysales = train %>% group_by(Outlet_Identifier) %>% 
  summarise_at(vars(Item_Outlet_Sales), funs(Count = n(),Sales_Mean = mean))

# Outlet_Establishment_Year
outletyear_bysales = train %>% group_by(Outlet_Establishment_Year) %>% 
  summarise_at(vars(Item_Outlet_Sales), funs(Count = n(),Sales_Mean = mean))

# Bivariate analysis of categorical variables

# Visualizing each categorical data table with mean and count seperately

# Visualing Item_Fat_Content with Mean of Item_Outlet_Sales
j1 <- ggplot(fat_bysales, aes(Item_Fat_Content, Sales_Mean)) + 
  geom_bar(stat='summary', fun.y='mean',fill='slateblue1',color='black') + 
  geom_point() + 
  geom_text(aes(label = ceiling(Sales_Mean)), vjust=-0.3, size=3.5) + 
  ggtitle("Item Fat Content Vs Sales") + 
  theme(plot.title = element_text(hjust = 0.5))

# Visualing Item_Fat_Content with Count of Item_Outlet_Sales
j2 <- ggplot(fat_bysales, aes(Item_Fat_Content, Count)) + 
  geom_bar(stat='summary', fun.y='mean',fill='slateblue1',color='black') + 
  geom_point() + 
  geom_text(aes(label = ceiling(Count)), vjust=-0.3, size=3.5) + 
  ggtitle("Item Fat Content Vs Sales") + 
  theme(plot.title = element_text(hjust = 0.5))

# Visualing Item_Type with Mean of Item_Outlet_Sales
j3 <- ggplot(type_bysales, aes(Item_Type, Sales_Mean)) + 
  geom_bar(stat='summary', fun.y='mean',fill='slateblue1',color='black') + 
  geom_point() + 
  geom_text(aes(label = ceiling(Sales_Mean)), vjust=-0.3, size=3.5) + 
  ggtitle("Item Type Vs Sales") + 
  theme(plot.title = element_text(hjust = 0.5)) + 
  theme(axis.text.x = element_text(angle=45,vjust=0.5))

graph <- plot_grid(j1,j2,ncol=2)
plot_grid(graph,j3,nrow=2)

# Visualing Item_Type with Count of Item_Outlet_Sales
j4 <- ggplot(type_bysales, aes(Item_Type, Count)) + 
  geom_bar(stat='summary', fun.y='mean',fill='slateblue1',color='black') + 
  geom_point() + 
  geom_text(aes(label = ceiling(Count)), vjust=-0.3, size=3.5) + 
  ggtitle("Item Type Vs Sales") + 
  theme(plot.title = element_text(hjust = 0.5)) + 
  theme(axis.text.x = element_text(angle=45,vjust=0.5))

# Visualing Outlet_Size with Mean of Item_Outlet_Sales
j5 <- ggplot(outletsize_bysales, aes(Outlet_Size, Sales_Mean)) + 
  geom_bar(stat='summary', fun.y='mean',fill='slateblue1',color='black') + 
  geom_point() + 
  geom_text(aes(label = ceiling(Sales_Mean)), vjust=-0.3, size=3.5) + 
  ggtitle("Outlet Size Vs Sales") + 
  theme(plot.title = element_text(hjust = 0.5))

plot_grid(j4,j5,nrow=2)

# Visualing Outlet_Location_Type with Mean of Item_Outlet_Sales
j6 <- ggplot(outletlocation_bysales, aes(Outlet_Location_Type, Sales_Mean)) + 
  geom_bar(stat='summary', fun.y='mean',fill='slateblue1',color='black') + 
  geom_point() + 
  geom_text(aes(label = ceiling(Sales_Mean)), vjust=-0.3, size=3.5) + 
  ggtitle("Outlet Location Type Vs Sales") + 
  theme(plot.title = element_text(hjust = 0.5))

# Visualing Outlet_Type with Mean of Item_Outlet_Sales
j7 <- ggplot(outlettype_bysales, aes(Outlet_Type, Sales_Mean)) + 
  geom_bar(stat='summary', fun.y='mean',fill='slateblue1',color='black') + 
  geom_point() + 
  geom_text(aes(label = ceiling(Sales_Mean)), vjust=-0.3, size=3.5) + 
  ggtitle("Outlet Type Vs Sales") + 
  theme(plot.title = element_text(hjust = 0.5)) + 
  theme(axis.text.x = element_text(angle=45,vjust=0.5))

plot_grid(j6,j7,nrow=2)

# Visualing Outlet_Identifier with Mean of Item_Outlet_Sales
j8 <- ggplot(outletid_bysales, aes(Outlet_Identifier, Sales_Mean)) + 
  geom_bar(stat='summary', fun.y='mean',fill='slateblue1',color='black') + 
  geom_point() + 
  geom_text(aes(label = ceiling(Sales_Mean)), vjust=-0.3, size=3.5) + 
  ggtitle("Outlet Identifier Vs Sales") + 
  theme(plot.title = element_text(hjust = 0.5)) + 
  theme(axis.text.x = element_text(angle=45,vjust=0.5))

# Visualing Outlet_Establishment_Year with Mean of Item_Outlet_Sales
j9 <- ggplot(outletyear_bysales, aes(factor(Outlet_Establishment_Year), Sales_Mean)) + 
  geom_bar(stat='summary', fun.y='mean',fill='slateblue1',color='black') + 
  geom_point() + 
  geom_text(aes(label = ceiling(Sales_Mean)), vjust=-0.3, size=3.5) + 
  xlab('Outlet_Establishment_Year') + 
  ggtitle("Outlet Establishment Year Vs Sales") + 
  theme(plot.title = element_text(hjust = 0.5))

plot_grid(j8,j9,nrow=2)

# Creating a violin graph for each categorical variable w.r.t distribution of Item_Outlet_Sales
g1 <- train %>% ggplot(aes(x=Item_Fat_Content, y=Item_Outlet_Sales)) + 
  geom_violin(fill='midnightblue') +
  ggtitle('Fat Content Vs Sales') +
  theme(plot.title = element_text(hjust = 0.5))

g2 <- train %>% ggplot(aes(x=Item_Type, y=Item_Outlet_Sales)) +
  geom_violin(fill='midnightblue') +
  theme(axis.text.x = element_text(angle=45,vjust=0.5)) +
  ggtitle('Item Type Vs Sales') +
  theme(plot.title = element_text(hjust = 0.5))

plot_grid(g1, g2, nrow = 2)

g3 <- train %>% ggplot(aes(x=Outlet_Size, y=Item_Outlet_Sales)) + 
  geom_violin(fill='midnightblue') +
  ggtitle('Outlet Size Vs Sales') +
  theme(plot.title = element_text(hjust = 0.5))

g4 <- train %>% ggplot(aes(x=Outlet_Location_Type, y=Item_Outlet_Sales)) + 
  geom_violin(fill='midnightblue') +
  ggtitle('Location Type Vs Sales') +
  theme(plot.title = element_text(hjust = 0.5))

plot_grid(g3, g4, nrow = 2)

g5 <- train %>% ggplot(aes(x=Outlet_Type, y=Item_Outlet_Sales)) + 
  geom_violin(fill='midnightblue') + 
  theme(axis.text.x = element_text(angle=45,vjust=0.5)) + 
  ggtitle('Outlet Type Vs Sales') +
  theme(plot.title = element_text(hjust = 0.5))

g6 <- train %>% ggplot(aes(x=Outlet_Identifier, y=Item_Outlet_Sales)) + 
  geom_violin(fill='midnightblue') + 
  theme(axis.text.x = element_text(angle=45,vjust=0.5)) +
  ggtitle('Outlet Identifier Vs Sales') +
  theme(plot.title = element_text(hjust = 0.5))

plot_grid(g5, g6, nrow = 2)

#Multivariate Analysis

# Outlet_Location_Type vs Outlet_Type vs Item_Outlet_Sales
a <- train %>% group_by(Outlet_Location_Type, Outlet_Type) %>% 
  summarise_at(vars(Item_Outlet_Sales), funs(Sales_Count = n()))

ggplot(a,aes(Outlet_Location_Type,Sales_Count,fill=Outlet_Type)) + geom_bar(stat='identity')

# Item_Visibility vs Item_MRP vs Item_Type
ggplot(train,aes(x=Item_Visibility,y=Item_MRP)) + 
  geom_point(aes(color=Item_Type)) + 
  theme_bw()+facet_wrap(~Item_Type) + 
  ggtitle('Item Type Visibility Vs MRP') + 
  theme(plot.title = element_text(hjust = 0.5))

# Creating a heatmap
ggplot(train, aes(Outlet_Identifier, Item_Type)) +
  geom_raster(aes(fill = Item_MRP)) + 
  ggtitle('Outlet Identifier Vs Item Wise MRP') + 
  theme(plot.title = element_text(hjust = 0.5))

# Building the model

# Getting ready the model data
xtrain = train %>% dplyr::select(-Item_Identifier)
# To check whether it contains the NA or not
sum(is.na.data.frame(xtrain))

# Omitting the NA Values
xtrain = na.omit(xtrain)

# Creating a linear Model with Item_Outlet_Sales as target variable
linear_model = lm(Item_Outlet_Sales ~ ., data=xtrain, na.action = na.omit)

# Summarizing the model
summary(linear_model)

# Plotting the train model
plot(linear_model)

# Determining AIC and BIC Values
AIC(linear_model)
BIC(linear_model)

# Segregating the data into train and test (train 80%, test 20%)
trainingRowIndex <- sample(1:nrow(xtrain), 0.8*nrow(xtrain))
trainingData <- xtrain[trainingRowIndex, ]
testData  <- xtrain[-trainingRowIndex, ]  

# Building a model in train data
lmMod <- lm(Item_Outlet_Sales ~ ., data=trainingData, na.action = na.omit)

# Predicting the test data
distPred <- predict(lmMod, data=testData)

# Summarizing the model
summary(lmMod)

# Comparing actual values and predicted values
actuals_preds <- data.frame(cbind(actuals=testData$Item_Outlet_Sales, predicted=distPred))
correlation_accuracy <- cor(actuals_preds)
head(actuals_preds)

# Determining the RMSE Value of Model
mse_lm <- mse(distPred, testData$Item_Outlet_Sales)
cat('Linear model MSE: ', sqrt(mse_lm))
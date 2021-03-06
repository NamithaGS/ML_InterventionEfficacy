setwd("C:/Users/Namithaa/Desktop/DR/project/models")
dataset1 <- read.csv('Merged_Data_Filtered_Columns_cat_asfactors1.csv')
dataset1 <- na.omit(dataset1)
dataset1 <- dataset1[-c(46,48)]
#dataset1$mean_effectsize <- as.numeric(dataset1$mean_effectsize)

#range01 <- function(x){(x-min(x))/(max(x)-min(x))}
#dataset1$mean_rxbaseline <- range01(dataset1$mean_rxbaseline)
range01 <- function(x){ log(x  + 1 -min(x))}
dataset1$mean_effectsize <- range01(dataset1$mean_effectsize)

# Create partitions
library("magrittr")
library("caret")
library(randomForest)
library(recommenderlab)
library(betareg)
library(pROC)
library(nnet)
library(ROCR)
library(reshape)
library(devtools)
source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')


set.seed(300)
inTraining <- createDataPartition(dataset1$mean_effectsize, p = .80, list = FALSE)

# Converting to factors

dataset1$ap <- factor(dataset1$ap)
dataset1$ja <- factor(dataset1$ja)
dataset1$sn <- factor(dataset1$sn)
dataset1$apja <- factor(dataset1$apja)
dataset1$finished <- factor(dataset1$finished)
dataset1$sex <- factor(dataset1$sex)
dataset1$epc_withdrawn <- factor(dataset1$epc_withdrawn)
dataset1$epc_randomization_unit_has_resid <- factor(dataset1$epc_randomization_unit_has_resid)
dataset1$epc_randomization_unit_enrolled_ <- factor(dataset1$epc_randomization_unit_enrolled_)
dataset1$provid <- factor(dataset1$provid)
dataset1$institution <- factor(dataset1$institution)
dataset1$randomization_unit_id <- factor(dataset1$randomization_unit_id)
dataset1$provcred <- factor(dataset1$provcred)
dataset1$boardcert_training <- factor(dataset1$boardcert_training)
dataset1$epc_foreigncred <- factor(dataset1$epc_foreigncred)
dataset1$epc_randomization_unit_ehr_imple <- factor(dataset1$epc_randomization_unit_ehr_imple)
#dataset1$mean_rxbaseline <- factor(dataset1$mean_rxbaseline)


dataset1 <- dataset1[-c(1,35,36,37)] # Don't need ID for building model, raw baseline, raw interv and provid

# Split into sets
training <- dataset1[ inTraining,]
testing  <- dataset1[-inTraining,]

# Variable for cross-validation 10 fold

fitControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  ## repeated ten times
  repeats = 10)

# Feature selection 
# define the control using a random forest selection function

control <- rfeControl(functions=rfFuncs, method="cv", number=10)

# Performing feature selection

set.seed(800)
results <- rfe(training[,1:41], training[,42], sizes=c(1:41), rfeControl=control)
# summarize the results
print(results)
# list the chosen features
predictors(results)
# plot the results
plot(results, type=c("g", "o"))

# Building model

truth_values = training$mean_effectsize
#scale(training$mean_effectsize)
#model <- glm(mean_effectsize~bronchitis_pct3+nos_pct3+sinusitis_pct2+sinusitis_pct3+age, family = binomial (link='logit'),data=training)
#model  <- glm(mean_effectsize~. , family = binomial (link='logit'),data=training)
#randomization_unit_id+ sn+ guideline_agree1_nos+ sex+ guideline_agree2_nos

#75: randomization_unit_id+ sn+ guideline_agree2_sinusitis+ epc_baseline_fte+ bronchitis_pct3

#mean_rxbaseline+epc_randomization_unit_ehr_imple+nos_pct1+epc_randomization_unit_enrolled_+epc_randomization_unit_has_resid
set.seed(800)
model  <- train(mean_effectsize~mean_rxbaseline+randomization_unit_id+nos_pct3+mean_rxpost_int+guideline_agree1_pharyngitis, family = binomial ,data=training
                ,method="glm")
model


#training
model_prob1 = predict(model)
model_prob = predict(model,type="response")
training = training %>% mutate(model_pred=1*(model_prob>0.5))
xtab = table(training$high_effectsize,training$model_pred)
confusionMatrix(xtab)


plot( training$high_effectsize,training$model_pred)
#testing

model_prob = predict(model,newdata=testing,type="response",se.fit=FALSE)
testing = testing %>% mutate(model_pred=1*(model_prob>0.5))
xtab = table(testing$high_effectsize,testing$model_pred)
confusionMatrix(xtab)
plot (confusionMatrix())

# lmfit = lm(mean_effectsize ~ randomization_unit_id+nos_pct3+ bronchitis_pct3+ sinusitis_pct3+ epc_intervention_fte  , data = training
#            )
# coef(lmfit)
# lmmiset.seed(825)
# truth = testing$mean_effectsize
# pred = predict(lmfit,testing)
# xtab = table(pred,truth)
# confusionMatrix(xtab)
# actuals_preds <- data.frame(cbind(actuals=testing$mean_effectsize, predicteds=pred))

gbmFit1 <- train(mean_effectsize ~ epc_randomization_unit_enrolled_ + epc_randomization_unit_has_resid + nos_pct1 + guideline_agree1_nos  , data = training,
                 method = "gbm",
                 trControl = fitControl,
                 ## This last option is actually one
                 ## for gbm() that passes through
                 verbose = FALSE)

C5Fit1 <- train(mean_effectsize ~ epc_randomization_unit_enrolled_ + epc_randomization_unit_has_resid + nos_pct1 + guideline_agree1_nos  , data = training,
                method = "C5.0",
                trControl = fitControl,
                ## This last option is actually one
                ## for gbm() that passes through
                verbose = FALSE)

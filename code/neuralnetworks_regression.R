setwd("C:/Users/Namithaa/Desktop/DR/project/models")
dataset1 <- read.csv('Merged_Data_Filtered_Columns_cat_asfactors1.csv')
dataset1 <- na.omit(dataset1)
dataset1 <- dataset1[-c(46,48)]
#dataset1$high_effectsize <- factor(dataset1$high_effectsize)

#scale(dataset1$mean_effectsize)
 # range01 <- function(x){(x-min(x))/(max(x)-min(x))}
 # dataset1$mean_effectsize <- range01(dataset1$mean_effectsize)

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

truth_values_training = training$mean_effectsize
truth_values_testing = testing$mean_effectsize
#scale(training$mean_effectsize)
#model <- glm(mean_effectsize~bronchitis_pct3+nos_pct3+sinusitis_pct2+sinusitis_pct3+age, family = binomial (link='logit'),data=training)
#model  <- glm(mean_effectsize~. , family = binomial (link='logit'),data=training)

#mean_rxbaseline + epc_randomization_unit_ehr_imple + nos_pct1 + epc_randomization_unit_enrolled_ + epc_randomization_unit_has_resid

set.seed(800)
#n <- names(training)
#f <- as.formula(paste("high_effectsize ~", paste(n[!n %in% "high_effectsize"], collapse = " + ")))
#nn <- neuralnet(f,data=training,hidden=c(5,3),linear.output=F,act.fct = "logistic")


#Regression
#nn <- nnet(mean_effectsize~mean_rxbaseline + randomization_unit_id + nos_pct3 + bronchitis_pct3 + guideline_agree2_nos,data=training,hidden=c(5,3), size=15,linear.output=T,maxiter= 100, act.fct = "logistic" )
#nn <- nnet(mean_effectsize~mean_rxbaseline + randomization_unit_id + nos_pct3 + bronchitis_pct3 + guideline_agree2_nos,data=training,size=3,linear.output=T,maxiter= 1000,decay=1e-1)

nn <- train(mean_effectsize~mean_rxbaseline + randomization_unit_id + nos_pct3 + bronchitis_pct3 + guideline_agree2_nos,data=training,maxiter= 300, method="nnet",linear.output=T, act.fct = "logistic" )
plot(nn)

nntesting <- train(mean_effectsize~mean_rxbaseline + randomization_unit_id + nos_pct3 + bronchitis_pct3 + guideline_agree2_nos,data=testing,maxiter= 300, method="nnet",linear.output=T, act.fct = "logistic" )
nntesting
plot(nntesting)


nnbest <- nnet(mean_effectsize~mean_rxbaseline + randomization_unit_id + nos_pct3 + bronchitis_pct3 + guideline_agree2_nos,data=training,size=5,linear.output=T,maxiter= 300,decay=0.1, act.fct = "logistic" )
plot(nnbest)

#regression MSE error training:
predictedvaluestraining = predict(nnbest,training,type="raw")
predictedvaluestraining
plot(truth_values_training,predictedvaluestraining )
abline(a=0,b=1)

predictedvaluestesting = predict(nnbest,testing,type="raw")
predictedvalues
plot(truth_values_testing,predictedvaluestesting )


xtab = table(truth_values_training,predictedvalues)
confusionMatrix(xtab)


#regression MSE error testing:



#d <- model.matrix( ~ high_effectsize +mean_rxbaseline + epc_randomization_unit_ehr_imple + nos_pct1 + epc_randomization_unit_enrolled_ + epc_randomization_unit_has_resid, data = training)
#n = colnames(d)
#n <- n[-c(1)]
#f <- as.formula(paste("high_effectsize1 ~", paste(n[!n %in% "high_effectsize1"], collapse = " + ")))
#nn <- nnet(f,data=d,hidden=c(5,3),linear.output=F,act.fct = "logistic" )
#nn <- nnet(high_effectsize~mean_rxbaseline + epc_randomization_unit_ehr_imple + nos_pct1 + epc_randomization_unit_enrolled_ + epc_randomization_unit_has_resid,data=training,hidden=c(5,3), size=15,linear.output=F,act.fct = "logistic" )
lol = class.ind(training$high_effectsize)
training$high_effectsize=NULL
nn <- nnet(lol ~ mean_rxbaseline + epc_randomization_unit_ehr_imple + nos_pct1 + epc_randomization_unit_enrolled_ + epc_randomization_unit_has_resid,data=training, size=30,linear.output=F,softmax=TRUE,act.fct = "logistic" ,entropy=TRUE,maxit=300)


#classification accuracy training
predictedclasses = predict(nn,training,type="class")
xtab = table(truth_values_training,predictedclasses)
confusionMatrix(xtab)

#classification accuracy testing
predictedclasses = predict(nn,testing,type="class")
xtab = table(truth_values_testing,predictedclasses)
confusionMatrix(xtab)

pred = ROCR::prediction( predict(nn,newdata=testing,type="raw"),testing$high_effectsize)
perf = performance(pred,"tpr","fpr")
plot(perf,lwd=2,col="blue",main="ROC - Neural Network on Adult")
abline(a=0,b=1)




testingdata = d <- model.matrix( ~ mean_rxbaseline + epc_randomization_unit_ehr_imple + nos_pct1 + epc_randomization_unit_enrolled_ + epc_randomization_unit_has_resid, data = testing)
pr.nn <- compute(nn,testingdata[,2:19])

pr.nn
val <- prediction(nn,testingdata[,2:19])



model  <- glm(high_effectsize~randomization_unit_id+ sn+ guideline_agree2_sinusitis+ epc_baseline_fte+ bronchitis_pct3, family = binomial (link='logit') ,data=training)

#training
model_prob = predict(model,type="response")
training = training %>% mutate(model_pred=1*(model_prob>0.5))
xtab = table(training$high_effectsize,training$model_pred)
confusionMatrix(xtab)

#testing

# model_prob = predict(model,newdata=testing,type="response")
# testing = testing %>% mutate(model_pred=1*(model_prob>0.5))
# xtab = table(testing$high_effectsize,testing$model_pred)
# confusionMatrix(xtab)
# plot (confusionMatrix())

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

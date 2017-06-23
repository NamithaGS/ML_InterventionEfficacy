setwd("C:/Users/Namithaa/Desktop/DR/project/models")
dataset1 <- read.csv('Merged_Data_Filtered_Columns_cat_asfactors1.csv')
dataset1 <- na.omit(dataset1)
dataset1 <- dataset1[-c(45,46,47)]
dataset1$high_effectsize <- factor(dataset1$high_effectsize)

# Create partitions

library(caret)
set.seed(300)
inTraining <- createDataPartition(dataset1$high_effectsize, p = .80, list = FALSE)

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

set.seed(500)
results <- rfe(training[,1:40], training[,41], sizes=c(1:40), rfeControl=control)
# summarize the results
print(results)
# list the chosen features
predictors(results)
# plot the results
plot(results, type=c("g", "o"))

# Building model

set.seed(825)
gbmFit1 <- train(high_effectsize ~ epc_randomization_unit_enrolled_ + epc_randomization_unit_has_resid + nos_pct1 + guideline_agree1_nos  , data = training, 
                 method = "gbm", 
                 trControl = fitControl,
                 ## This last option is actually one
                 ## for gbm() that passes through
                 verbose = FALSE)
set.seed(825)
C5Fit1 <- train(high_effectsize ~ epc_randomization_unit_enrolled_ + epc_randomization_unit_has_resid + nos_pct1 + guideline_agree1_nos  , data = training, 
                method = "C5.0", 
                trControl = fitControl,
                ## This last option is actually one
                ## for gbm() that passes through
                verbose = FALSE)

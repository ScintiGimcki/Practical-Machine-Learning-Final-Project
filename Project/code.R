pml_training <- read.csv("pml-training.csv", stringsAsFactors = FALSE, na.strings = "")
pml_training$classe <- factor(pml_training$classe)
pml_training[,-c(1:6, 160)] <- apply(pml_training[,-c(1:6, 160)], 2, as.numeric)
pml_testing <- read.csv("pml-testing.csv", stringsAsFactors = FALSE, na.strings = "")
pml_testing[,-c(1:6, 160)] <- apply(pml_testing[,-c(1:6, 160)], 2, as.numeric)

library(caret)
## -- omit time
pml_training <- pml_training[,-c(1:5)]

## -- check and fix na
check_na <- function(dataframe, benchmark){
  naPerc <- as.vector(apply(dataframe, 2, function(x){mean(is.na(x))}))
  which(naPerc > benchmark)
}
naArray <- check_na(pml_training, 0.1)
lowNaTrain <- pml_training[,-naArray]
which(as.vector(apply(lowNaTrain, 2, function(x){mean(is.na(x))}))>0)

## check correlation
check_num <- function(dataframe){
  classArray <- rep("", dim(dataframe)[2])
  for (i in 1:length(names(dataframe))) {
    classArray[i] <- class(dataframe[[names(dataframe)[i]]])
  }
  which(classArray == "numeric")
}
numVar <- check_num(lowNaTrain)
corM <- abs(cor(lowNaTrain[, numVar]))
diag(corM) <- 0
largeCor <- as.data.frame(which(corM > 0.8, arr.ind = T))
largeCorIndex <- unique(largeCor$row)
numTrain <- lowNaTrain[, numVar]
scaleProc <- preProcess(numTrain, method = c("center", "scale"))
numTrain <- predict(scaleProc, numTrain)
pcaProc <- preProcess(numTrain[,largeCorIndex], method = c("pca"), thresh = 0.9)
pcaVar <- predict(pcaProc, numTrain[, largeCorIndex])

## construct Training and Testing set, and construct cross validation
pcaTrain <- cbind(pml_training$classe, pcaVar, numTrain[, -largeCorIndex], lowNaTrain[, -numVar])
names(pcaTrain)[1] <- "classe"
inTrain <- createDataPartition(pml_training$classe, p=0.7, list = FALSE)
training <- pcaTrain[inTrain,]
validation <- pcaTrain[-inTrain,]
scaleTesting <- predict(scaleProc, pml_testing)
testing <- predict(pcaProc, scaleTesting)

## training
library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 1) 
registerDoParallel(cluster)
fitcontrol <- trainControl(method = "cv", number = 5, allowParallel = TRUE)
modelRF <- train(classe~., data = training, method = "rf", trControl = fitcontrol)
modelGBM <- train(classe~., data = training, method = "gbm")
modelLDA <- train (classe~., data = training, method = "lda")
stopCluster(cluster)
registerDoSEQ()

predictionRF <- mean(predict(modelRF, validation) == validation$classe)
predictionGBM <- mean(predict(modelGBM, validation) == validation$classe)
predictionLDA <- mean(predict(modelLDA, validation) == validation$classe)

predict(modelRF, testing)
# B A B A A E D B A A B C B A E E A B B B

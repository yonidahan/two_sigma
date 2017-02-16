
#two_sigma_xgb0

lapply(c("xgboost","mlr","dtplyr","caret","parallel","parallelMap","data.table"),
       require,character.only=T)

setwd("C:/Users/Sarah/Documents/Data Science/Projects/Two Sigma")

train<-read.csv("train_r.csv",stringsAsFactors = F)
train<-train[,!names(train)%in%c("id","listing_id")]

labels<-as.numeric(as.factor(train$interest_level))-1

train_matrix<-model.matrix(~.+0,data = train[,!names(train)%in%c("interest_level")])

train_index <- createDataPartition(train$interest_level, p = .7, list = FALSE)

train_tr<-train_matrix[train_index,]
train_ts<-train_matrix[-train_index,]

labels_tr<-labels[train_index]
labels_ts<-labels[-train_index]


dtrain <- xgb.DMatrix(data = train_tr,label = labels_tr)
dtest <- xgb.DMatrix(data = train_ts,label = labels_ts)

params <- list(
  booster = "gbtree",
  objective = "multi:softprob",
  num_class = 3,
  eta=0.3,
  gamma=0,
  max_depth=6,
  min_child_weight=1,
  subsample=1,
  colsample_bytree=1
)

xgbcv <- xgb.cv(params = params
                ,data = dtrain
                ,nrounds = 500
                ,nfold = 5
                ,showsd = T
                ,stratified = T
                ,print.every.n = 10
                ,eval_metric = "mlogloss" 
                ,early.stop.round = 20
                ,maximize = F
)

min(xgbcv$test.mlogloss.mean)


#Use test dataset
xgb1 <- xgb.train(
  params = params
  ,data = dtrain
  ,nrounds = 178
  ,watchlist = list(val=dtest,train=dtrain)
  ,print.every.n = 10
  ,maximize = F
  ,eval_metric = "mlogloss"
  ,early.stop.round = 10
  
)

important <- xgb.importance(feature_names = colnames(train_tr),model = xgb1)


#random grid search procedure
fact_col <- colnames(train)[sapply(train,is.character)]
for(i in fact_col)
  set(train,j=i,value = factor(train[[i]]))

train<-createDummyFeatures(obj=train,target="interest_level")#One-Hot

train_index <- createDataPartition(train$interest_level, p = .7, list = FALSE)

train_tr<-train[train_index,]
train_ts<-train[-train_index,]

train_tr<-train_tr[1:5000,]

traintask <- makeClassifTask(data = train_tr,target = "interest_level")
testtask <- makeClassifTask(data = train_ts,target = "interest_level")

lrn <- makeLearner("classif.xgboost",predict.type = "response")

lrn$par.vals <- list(
  objective = "multi:softprob",
  eval_metric="mlogloss",
  nrounds=100L,
  eta=0.1
)

params <- makeParamSet(
  makeDiscreteParam("booster",values = c("gbtree","gblinear")),
  makeIntegerParam("max_depth",lower = 3L,upper = 10L),
  makeNumericParam("min_child_weight",lower = 1L,upper = 10L),
  makeNumericParam("subsample",lower = 0.5,upper = 1),
  makeNumericParam("colsample_bytree",lower = 0.5,upper = 1),
  makeNumericParam("gamma",lower = 5,upper = 10)
)

rdesc <- makeResampleDesc("CV",stratify = T,iters=5L)

ctrl <- makeTuneControlRandom(maxit = 10L)

#set parallel backend
library(parallel)
library(parallelMap)
parallelStartSocket(cpus = detectCores())

#parameter tuning
mytune <- tuneParams(learner = lrn
                     ,task = traintask
                     ,resampling = rdesc
                     ,measures = acc
                     ,par.set = params
                     ,control = ctrl
                     ,show.info = T)




#Two Sigma Connect: Rental Listing Inquiries
lapply(c("jsonlite","tm","magrittr","h2o","dplyr","lubridate",
         "syuzhet","DT"),
       require,character.only=T)

setwd("C:/Users/Sarah/Documents/Data Science/Projects/Two Sigma")



train<-read_json("train.json")

##train_df<-data.frame(do.call("cbind",train))

#Features
thresh_feat<-100#Arbitrary at this stage
features<-train$features
features[sapply(features,length)==0]<-NA
features<-lapply(features,data.frame,stringsAsFactors=F)
features<-rbind_all(features)

features[!is.na(features)]<-1
features[is.na(features)]<-0
features<-data.matrix(features)

feat_to_keep<-features%>%apply(2,sum)
feat_to_keep<-names(feat_to_keep[feat_to_keep>thresh_feat])

features<-as.data.frame(features[,colnames(features)%in%feat_to_keep])

n_features<-sapply(train$features,length)

display_address<-unlist(train$display_address)
street_address<-unlist(train$street_address)


description<-train$description%>%
  unlist%>%
  tolower%>%
  removePunctuation%>%
  removeWords(stopwords("english"))%>%
  gsub(pattern="br",replacement="")%>%
  gsub(pattern="[[:space:]]+[a-zA-Z]{1,2}[[:space:]]+",replacement="")%>%
  stripWhitespace

sentiment_descr<-get_nrc_sentiment(description)

n_descr<-sapply(description,
                FUN=function(x)sapply(gregexpr("\\W+", x), length) + 1
)

#Euclidean Distance to city center
ny_center_lat<-40.785091
ny_center_lon<--73.968285

center_distance <-
  mapply(function(long, lat) sqrt((long-ny_center_lon)^2  + (lat-ny_center_lat)^2),
         train$longitude,
         train$latitude) 



#Construct the DF
train_r<-data.frame(
  "id"=attr(train$bedrooms,"names"),
  "bathrooms"=as.numeric(unlist(train$bathrooms)),
  "bedrooms"=as.numeric(unlist(train$bedrooms)),
  "building_id"=as.integer(as.factor(unlist(train$building_id))),
  
  "month"=months(as.POSIXlt(strptime(unlist(train$created),
                                     "%Y-%m-%d %H:%M:%S"))),
  "weekday"=weekdays(as.POSIXlt(strptime(unlist(train$created),
                                     "%Y-%m-%d %H:%M:%S"))),
  "hour"=hour(as.POSIXlt(strptime(unlist(train$created),
                                  "%Y-%m-%d %H:%M:%S"))),
  
  "latitude"=unlist(train$latitude),
  "longitude"=unlist(train$longitude),
  "listing_id"=unlist(train$listing_id),
  "manager_id"=as.factor(unlist(train$manager_id)),
  "n_photos"=sapply(train$photos,length),
  "price"=as.numeric(train$price),
  "interest_level"=as.factor(unlist(train$interest_level)),
  "n_descr"=n_descr,
  "n_features"=n_features,
  "center_distance"=center_distance,
  "sentiment_descr"=sentiment_descr,
  "features"=features)

#df_path<-"C:/Users/Sarah/Documents/Data Science/Projects/Two Sigma/train_df.csv"
#write.csv(train_df,df_path,row.names=F)

h2o.init(nthreads = -1)

#train_df<-h2o.importFile(df_path,destination_frame = "train_df_hex")
train_df<-as.h2o(train_r,destination_frame="train_hex")
train_df<-h2o.asnumeric(train_df)
train_df$interest_level<-h2o.asfactor(train_df$interest_level)


feat_names<-setdiff(names(train_df),c("interest_level","id"))

#Split CV
train_split<-h2o.splitFrame(data = train_df,c(0.8),seed=2008)
train_df_tr<-train_split[[1]]
train_df_te<-train_split[[2]]

#GBM
gbm0<-h2o.gbm(x=feat_names,y="interest_level",
              training_frame = train_df_tr,validation_frame = train_df_te,
              model_id="gbm0",distribution="multinomial",
              nfolds=5,ntrees=200,learn_rate = 0.01
              ,max_depth = 7
              ,min_rows = 20
              ,sample_rate = 0.7
              ,col_sample_rate = 0.7
              ,stopping_rounds = 5
              ,stopping_metric = "logloss"
              ,stopping_tolerance = 0
              ,seed=2008)


#LM no regul
lm0<-h2o.glm(x=feat_names,y="interest_level",
             training_frame = train_df_tr,validation_frame = train_df_te,
             model_id="lm0",family="multinomial",
             nfolds=5,standardize=T,remove_collinear_columns = T)


#LM regul
lm_regul_grid1<-h2o.grid("glm",grid_id="lm_regul_grid1",
                         x=feat_names,y="interest_level",
                         training_frame=train_df_tr,
                         validation_frame=train_df_te,
                         family="multinomial",standardize=T,
                         hyper_params=list(alpha=c(0,0.25,0.5,0.75,1),
                                            lambda=c(10^-2,10^-3,10^-4,10^-5,10^-6,10^-7)),
                         search_criteria = list(strategy="RandomDiscrete",
                                                 stopping_metric="logloss"))

lm_regul_grid1_sort<- h2o.getGrid(grid_id = "lm_regul_grid1", sort_by = "logloss")
lm_regul1<-h2o.getModel(lm_regul_grid1_sort@model_ids[[1]])
h2o.performance(lm_regul1,valid=T)@metrics$logloss


dfs <- lapply(train$features, data.frame,stringsAsFactors=F)
pz<-rbind_all(dfs)
pz[!is.na(pz)]<-1
pz[is.na(pz)]<-0


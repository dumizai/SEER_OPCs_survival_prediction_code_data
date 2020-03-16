######################################################
########### R codes attached to article ##############
#### Application of machine learning algorithms for###
########## predicting the survial of #################
###########oral and pharyngeal cancers ###############
######################################################
########### corespondence to: Mi Du ##################
######################################################

setwd("")
library(readxl)
library(table1)
library(survival)
library(survivalROC)
library(survminer)
library(rms)
library(CoxBoost)
library(magicfor)
library(randomForestSRC)
library(pec)
library(mlr)

# read data and data preperation#
full_data<-read_xls('data.xls')
full_data$Sex<-as.factor(full_data$Sex)
full_data$Race<-as.factor(full_data$Race)
full_data$Marital_s<-as.factor(full_data$Marital_s)
full_data$T_n<-as.factor(full_data$T_n)
full_data$N_n<-as.factor(full_data$N_n)
full_data$M_n<-as.factor(full_data$M_n)
full_data$Stage<-as.factor(full_data$Stage)
full_data$Grade<-as.factor(full_data$Grade)
full_data$TS_n<-as.factor(full_data$TS_n)
full_data$Surgery<-as.factor(full_data$Surgery)
full_data$LN_r<-as.factor(full_data$LN_r)
full_data$ICD_n<-as.factor(full_data$ICD_n)

levels(full_data$ICD_n)<-c('Lip (C00)','Base of tounge (C01)', 'Other parts of tounge (C02)','Gum (C03)',
                           'Floor of mouth (C04)','Palate(C05)','Other oral cavity(C06)','Parotid gland (C07)',
                           'Other salivary glands (C08)','Tonsil (C09)','Oropharynx (C10)','Nasopharynx (C11)',
                           'Pyriform sinus (C12)','Hypopharynx (C13)','Others (C14)')
levels(full_data$T_n)<-c("T1" ,"T1", "T2", "T3", "T4", "TX")
# remove samples with survial time less than 1 month
full_data<-full_data[-which(full_data$Survival_m<=1),]
# extract data for three year cohort
threedata_full<-full_data[full_data$Three_d == 1,]
threedata_full<-as.data.frame(threedata_full[,c('Three_y_Death','Survival_m','Age','Sex', 'Race', 'Marital_s', 
                                                'Grade', 'T_n', 'N_n','M_n', 'Stage', 
                                                'LN_r','TS_n','Surgery', 'ICD_n')])
threedata <-na.omit(threedata_full)
threedata<-droplevels(threedata)

##########################################
#########Three year model##################

#cox model in complete cases
random_tune <- makeTuneControlRandom(maxit = 1L)
rdesc = makeResampleDesc( "CV", iters = 10, stratify = TRUE ) #"Holdout")

seeds<-c(1:50)
magic_for(print, silent = TRUE)
for (k in seeds){
  set.seed(k)
  index <- sample(1:nrow(threedata), round(0.3*nrow(threedata)))
  test_set <- threedata[-index,]
  train_set<- threedata[index,]
  task<-makeSurvTask(data = train_set,target=c('Survival_m','Three_y_Death'))
  cox.lrn <- makeLearner(cl="surv.coxph", 
                         predict.type="response")
  modcox = train(cox.lrn, task) 
  train_pred<-predict(modcox, newdata = train_set)
  train_p<-performance(train_pred, measures = list(cindex)) 
  test_pred<-predict(modcox, newdata = test_set)
  test_p<-performance(test_pred, measures = list(cindex)) 
  print(round(train_p,3),round(test_p,3))
}
performance<-magic_result_as_dataframe()
summary(performance)

# surv.rpart in complete cases
magic_for(print, silent = TRUE)
for (k in seeds){
  set.seed(k)
  index <- sample(1:nrow(threedata), round(0.3*nrow(threedata)))
  test_set <- threedata[-index,]
  train_set <- threedata[index,]
  task<-makeSurvTask(data = train_set,target=c('Survival_m','Three_y_Death'))
  rpart.lrn<-makeLearner(cl='surv.rpart', predict.type = 'response')
  getParamSet("surv.rpart")
  model_params_2<-makeParamSet(
    makeIntegerParam('minsplit', lower=1,upper=20),
    makeIntegerParam('maxdepth',lower=1,upper=30))
  # Tune model to find best performing parameter settings using random search algorithm
  tuned_model_2 <- tuneParams(learner = rpart.lrn,
                              task = task,
                              resampling = rdesc,
                              measures = cindex,       #  performance measure, this can be changed to one or many
                              par.set = model_params_2,
                              control = random_tune,
                              show.info = FALSE)
  # Apply optimal parameters to model
  rpart.lrn <- setHyperPars(learner = rpart.lrn,
                            par.vals = tuned_model_2$x)
  modrpart = train(rpart.lrn, task)
  train_pred_rpart<-predict(modrpart, newdata = train_set)
  train_p<-performance(train_pred_rpart, measures = list(cindex)) # c-index  in training set
  test_pred_rpart<-predict(modrpart, newdata = test_set) #prediction in test set
  test_p<-performance(test_pred_rpart, measures = list(cindex)) # 
  print(round(train_p,3),round(test_p,3))
}
performance<-magic_result_as_dataframe()
summary(performance)


#surv.randomForestSRC in complete cases
magic_for(print, silent = TRUE)
for (k in seeds) {
  set.seed(k)
  index <- sample(1:nrow(threedata), round(0.3*nrow(threedata)))
  train_set <- threedata[index,]
  test_set <- threedata[-index,]
  write.csv(train_set,'three_train_0.csv')
  write.csv(test_set,'three_test_0.csv')
  train_set<-read.csv('three_train_0.csv',header = T)
  test_set<-read.csv('three_test_0.csv',header = T)
  train_set<-train_set[,c(2:16)]
  test_set<-test_set[,c(2:16)]
  common <- intersect(names(train_set), names(test_set)) 
  for (p in common) { 
    if (class(train_set[[p]]) == "factor") { 
      levels(test_set[[p]]) <- levels(train_set[[p]]) 
      levels(train_set[[p]]) <- levels(test_set[[p]]) 
      # print(levels(test_set[[p]]))
    } 
  }
  task<-makeSurvTask(data = train_set,
                     target=c('Survival_m','Three_y_Death'))
  rfsrc.lrn<-makeLearner(cl='surv.randomForestSRC',
                         predict.type = 'response')
  getParamSet("surv.randomForestSRC")
  model_params_3<-makeParamSet(
    makeIntegerParam('ntree', lower=1000,upper=2000),
    makeIntegerParam('mtry',lower = 1,upper = 12),
    #makeIntegerParam('nodedepth',lower = 1, upper = 20),
    #makeIntegerParam('nodesize',lower = 3, upper=100),
    makeIntegerParam('nsplit',lower = 0, upper=20),
    makeDiscreteParam('splitrule',values = 'logrank', special.vals = list('logrank','logrankscore','random'))
  )
  # Tune model to find best performing parameter settings using random search algorithm
  tuned_model_3 <- tuneParams(learner = rfsrc.lrn,
                              task = task,
                              resampling = rdesc,
                              measures =  cindex,       #  performance measure, this can be changed to one or many
                              par.set = model_params_3,
                              control = random_tune,
                              show.info = FALSE)
  # Apply optimal parameters to model
  rfsrc.lrn <- setHyperPars(learner = rfsrc.lrn,
                            par.vals = tuned_model_3$x)
  modrfsrc = train(rfsrc.lrn, task)
  train_pred_rfsrc<-predict(modrfsrc, newdata = train_set)
  train_p<-performance(train_pred_rfsrc, measures = list(cindex)) # c-index  in training set
  test_pred_rfsrc<-predict(modrfsrc, newdata = test_set) #prediction in test set
  test_p<-performance(test_pred_rfsrc, measures = list(cindex)) #  
  print(round(train_p,3),round(test_p,3))
}
performance<-magic_result_as_dataframe()
summary(performance)


############## MI ############  
## impute missing data for three year cohort using random forest ##

f <- as.formula(Surv(Survival_m, Three_y_Death) ~ .)

threedata_MI <- randomForestSRC::impute.rfsrc(f, data=threedata_full, 
                                              splitrule = "random", 
                                              na.action='na.impute',
                                              nimpute = 5)

# to develop cox/tree/Rf in MI data for three year cohort, we need to
# replace complete cases with imputed data.



############pec###############
# calibration plot for three year cohort (complete cases, training set)
library(pec)
set.seed(19910220)
index1 <- sample(1:nrow(threedata), round(0.8*nrow(threedata)))
train_set_1 <- threedata[index1,]
test_set_1 <- threedata[-index1,]
cox1 <- coxph(Surv(Survival_m, Three_y_Death) ~., x = T, y = T, 
              data = train_set_1)
tree1 <- pecCtree(Surv(Survival_m, Three_y_Death)~.,
                  data=train_set_1)
rsf1 <- rfsrc(Surv(Survival_m, Three_y_Death)~.,
              data=train_set_1,
              ntree=1500,forest=TRUE,
              tree.err=T, importance=T,
              na.action = "na.impute")
cf1=calPlot(list(cox1, tree1,  rsf1),
            col=c("black",'red','green'),
            time=36,
            type="survival",
            legend=F, 
            data = train_set_1,
            splitMethod = "cv",
            B=10)
legend("topleft", legend=c("Cox","Tree","randomForestSRC"),
       col=c("black",'red','green'), lty=1, cex=0.8)

# overtime cindex using pec package ###
# overtime C index in complete cases for three year cohort (training set)
ApparrentCindex1 <- pec::cindex(list(cox1,                    
                                     tree1,
                                     rsf1),
                                formula=Surv(Survival_m, Three_y_Death) ~ .,
                                data=train_set_1,
                                splitMethod="cv",
                                B=10,
                                eval.times=seq(0,36,1))
plot(ApparrentCindex1,legend=c(2,1),xlim=c(0,36))
legend("topright", legend=c("Cox","Tree","randomForestSRC"),
       col=c("black",'red','green'), lty=1, cex=0.8)



####time consumption###
setseed(19910220)
task<-makeSurvTask(data = train_set_1,
                   target=c('Survival_m','Five_y_Death'))
lrns = list(makeLearner("surv.coxph"),
            makeLearner("surv.rpart"),
            makeLearner("surv.randomForestSRC"))
bmr = benchmark(lrns, task, rdesc, timetrain,show.info = FALSE)
perf = getBMRPerformances(bmr, as.df = TRUE)
sum(perf$timetrain[1:10]) # traning time for one iteration for Cox
sum(perf$timetrain[11:20])# traning time for one iteration for Tree
sum(perf$timetrain[21:30])# traning time for one iteration for RF

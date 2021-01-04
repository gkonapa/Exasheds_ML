rm(list = ls())
library(tidyverse)
library(DALEX)
library(mlr)
library(dplyr)
library(rgdal)
library(mlrMBO)
library(readr)
library(randomForest)
library(caret)
library(imbalance)
library(c2c)
library(ALEPlot)
library(Metrics)
get_legend<-function(myggplot){
  tmp <- ggplot_gtable(ggplot_build(myggplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)
}
camels_topo <- read_delim("./camels_attributes_v2.0/camels_topo.txt",  ";", escape_double = FALSE, trim_ws = TRUE)
myvars = c("gauge_id","elev_mean","slope_mean")
camels_topo <-camels_topo[myvars]
camels_clim <- read_delim("./camels_attributes_v2.0/camels_clim.txt",  ";", escape_double = FALSE, trim_ws = TRUE)
camels_hydro <- read_delim("./camels_attributes_v2.0/camels_hydro.txt",  ";", escape_double = FALSE, trim_ws = TRUE)
camels_soil <- read_delim("./camels_attributes_v2.0/camels_soil.txt",  ";", escape_double = FALSE, trim_ws = TRUE)
myvars <- names(camels_soil) %in% c("soil_porosity","soil_conductivity")
camels_soil = camels_soil[!myvars]
camels_vege <- read_delim("./camels_attributes_v2.0/camels_vege.txt",  ";", escape_double = FALSE, trim_ws = TRUE)
myvars <- names(camels_vege) %in% c("lai_diff","gvf_diff")
camels_vege = camels_vege[!myvars]
camels_geol <- read_delim("./camels_attributes_v2.0/camels_geol.txt",  ";", escape_double = FALSE, trim_ws = TRUE)
myvars <- names(camels_geol) %in% c("geol_2nd_class","glim_2nd_class_frac")
camels_geol = camels_geol[!myvars]
#camels_hfstats <- read.delim("E:/AI_challenge/hfstats.txt")
commondata=list(camels_clim, camels_vege, camels_soil,camels_topo,camels_hydro,camels_geol) %>% reduce(left_join, by = "gauge_id")
hyd_mod <- read.table("./NSE_hyd.txt", quote="\"", comment.char="")
LSTM <- read.csv("./NSE_pure_datadriven_all.txt", sep="")
hybrid1 <- read.csv("./NSE_hybrid1_all.txt", sep="")
hybrid2 <- read.csv("./NSE_hybrid2_all.txt", sep="")

DL_data <- cbind.data.frame(LSTM,hybrid1,hybrid2)
names(DL_data)=c("LSTM","HYBRID1","HYBRID2")
DL_data[DL_data<0]=-0.01
model_data <- rep(hyd_mod,3)
model_ful <- do.call(cbind.data.frame,model_data)
model_ful[model_ful<0]=-0.01
DL_data_diff <- (DL_data - model_ful)
basin_list <- read_csv("./basin_list.txt",
                       col_names = FALSE)
names(basin_list)=c("gauge_id")
#set.seed(71)
# k1<-kmeans(DL_data_diff$LSTM,3)
# centerx<-data.frame(k1$centers,1:3)
# names(centerx)=c("meank","cluster")
# classnse = data.frame(k1$cluster)
# classnse[classnse==centerx$cluster[which.max(centerx$meank)]] ="high"
# classnse[classnse==centerx$cluster[which.min(centerx$meank)]] ="low"
# idxhigh = which(classnse=="high")
# idxlow = which(classnse=="low")
# idxtotal = c(idxhigh,idxlow)
# idx = 1:531
# newidx=setdiff(idx,idxtotal)
# classnse[newidx,1]="mod"
# # classnse=data.frame(matrix(NA, nrow = 531, ncol = 0))
# # classnse$PERF[DL_data_diff$LSTM>0]="Good"
# # classnse$PERF[DL_data_diff$LSTM<0]="Bad"
ML_model <-cbind(basin_list,DL_data_diff$HYBRID1)
names(ML_model)=c("gauge_id","LSTM")
map_plot <-merge(ML_model,commondata,by="gauge_id")
positive_map = map_plot
positive_map[sapply(positive_map, is.character)] <- lapply(positive_map[sapply(positive_map, is.character)],
                                                           as.factor)
imp <- impute(positive_map,target = "LSTM", classes = list(factor = imputeMode(), integer = imputeMean(), numeric = imputeMean()))
imp_train <- imp$data
myvars <- names(imp_train) %in% c("gauge_id")
train_data <- imp_train[!myvars]
train_data <-na.omit(train_data)
train_data <- createDummyFeatures(
  train_data, target = "LSTM",
  cols = c(
    "high_prec_timing",
    "low_prec_timing",
    "dom_land_cover",
    "geol_1st_class"))
n_features <- length(setdiff(names(train_data), "LSTM"))
hyper_grid <- expand.grid(
  mtry = 1:10,
  nodesize = 1:12,
  replace = c(TRUE, FALSE))
for(i in seq_len(nrow(hyper_grid))) {
  # fit model for ith hyperparameter combination
  set.seed(12)
  fit <- randomForest(
    formula         = LSTM ~ .,
    data            = train_data,
    num.trees       = 500,
    mtry            = hyper_grid$mtry[i],
    nodesize        = hyper_grid$nodesize[i],
    replace         = hyper_grid$replace[i])
  # export OOB error
  new_data=subset(train_data, select=-c(LSTM))
  pred <- predict(fit, newdata=new_data)
  hyper_grid$rmse[i] <- rmse(train_data$LSTM,pred)
  hyper_grid$rs[i]<-cor(train_data$LSTM,pred)
  print(i)
  print(hyper_grid$rs[i])
}

set.seed(12)
fit <- randomForest(
  formula         = LSTM ~ .,
  data            = train_data,
  num.trees       = 500,
  mtry            = hyper_grid$mtry[which.min(hyper_grid$rmse)],
  nodesize        = hyper_grid$nodesize[which.min(hyper_grid$rmse)],
  replace         = hyper_grid$replace[which.min(hyper_grid$rmse)],
  importance=TRUE)

## first_imbalance
new_data=subset(train_data, select=-c(LSTM))
custom_predict <- function(object, newdata) {pred <- predict(object, newdata=newdata)
return(pred)}
explain_DL <- explain(model = fit,
                              data = new_data,
                              y = train_data$LSTM,
                              predict_function = custom_predict,
                              label = "Hybrid1")
rm(list=setdiff(ls(), c("explain_DL")))
library(ingredients)
fi_rf <- feature_importance(explain_DL)
camels_topo <- read_delim("./camels_attributes_v2.0/camels_topo.txt",  ";", escape_double = FALSE, trim_ws = TRUE)
myvars = c("gauge_id","elev_mean","slope_mean")
camels_topo <-camels_topo[myvars]
camels_clim <- read_delim("./camels_attributes_v2.0/camels_clim.txt",  ";", escape_double = FALSE, trim_ws = TRUE)
camels_hydro <- read_delim("./camels_attributes_v2.0/camels_hydro.txt",  ";", escape_double = FALSE, trim_ws = TRUE)
camels_soil <- read_delim("./camels_attributes_v2.0/camels_soil.txt",  ";", escape_double = FALSE, trim_ws = TRUE)
myvars <- names(camels_soil) %in% c("soil_porosity","soil_conductivity")
camels_soil = camels_soil[!myvars]
camels_vege <- read_delim("./camels_attributes_v2.0/camels_vege.txt",  ";", escape_double = FALSE, trim_ws = TRUE)
myvars <- names(camels_vege) %in% c("lai_diff","gvf_diff")
camels_vege = camels_vege[!myvars]
camels_geol <- read_delim("./camels_attributes_v2.0/camels_geol.txt",  ";", escape_double = FALSE, trim_ws = TRUE)
myvars <- names(camels_geol) %in% c("geol_2nd_class","glim_2nd_class_frac")
camels_geol = camels_geol[!myvars]
#camels_hfstats <- read.delim("E:/AI_challenge/hfstats.txt")
commondata=list(camels_clim, camels_vege, camels_soil,camels_topo,camels_hydro,camels_geol) %>% reduce(left_join, by = "gauge_id")
climate = cbind.data.frame(names(camels_clim),"Climate")
names(climate)=c("variable","type")
topography=cbind.data.frame(names(camels_topo),"Soil & Topography")
names(topography)=c("variable","type")
hydro=cbind.data.frame(names(camels_hydro),"Hydrology")
names(hydro)=c("variable","type")
soil=cbind.data.frame(names(camels_soil),"Soil & Topography")
names(soil)=c("variable","type")
vege=cbind.data.frame(names(camels_vege),"Vegetation")
names(vege)=c("variable","type")
geol=cbind.data.frame(names(camels_geol),"Geology")
names(geol)=c("variable","type")
catchments = rbind(climate,topography,hydro,soil,vege,geol)
sortedcharacters = sort(as.character(unique(fi_rf$variable)))
vimp_med=aggregate(dropout_loss~variable , data=fi_rf, FUN=mean)
names(vimp_med) =  c("variable","median")
vimp_max = aggregate(dropout_loss~variable , data=fi_rf, FUN=max)
names(vimp_max) =  c("variable","max")
vimp_min = aggregate(dropout_loss~variable , data=fi_rf, FUN=min)
names(vimp_min) =  c("variable","min")
vimp_DL = cbind(vimp_med,vimp_max$max,vimp_min$min)
names(vimp_DL) = c("variable","median","max","min")
vimp_DL <- merge(vimp_DL,catchments,by="variable",all = TRUE)
vimp_DL$type[vimp_DL$variable %in% sortedcharacters[23:34]]="Geology"
vimp_DL$type[vimp_DL$variable %in% sortedcharacters[7:19]]="Vegetation"
vimp_DL$type[vimp_DL$variable %in% sortedcharacters[42:45]]="Climate"
vimp_DL$type[vimp_DL$variable %in% sortedcharacters[51:54]]="Climate"
vimp_DL=na.omit(vimp_DL)
vimp_DL$model = "DL"
vimp_DL=droplevels.data.frame(vimp_DL)
a<-ggplot(data = vimp_DL,aes(x=variable,y=median,ymin = min, ymax= max))+
  geom_pointrange()+geom_point(color="red")+
  facet_grid(type~.,scales = "free",space = "free_y")+
  theme_bw()+theme(axis.text.x = element_text(angle = 90, hjust = 1,vjust = 0.1),text = element_text(size=15),panel.grid.major = element_blank(),
                   panel.grid.minor = element_blank())+
  labs(y="Drop out loss",x="Catchment Characteristics")+coord_flip()
topx=as.character(vimp_DL[order(vimp_DL$median,decreasing = TRUE),][1:10,1])
print(topx)
pp_DL<-conditional_dependency(explain_DL, variables =  topx)
pp_DL$`_vname_`<-factor(pp_DL$`_vname_`, levels = topx)

pp_DL$vars=as.character(pp_DL$`_vname_`)
unique(pp_DL$`_vname_`)
pp_DL$vars[pp_DL$vars=="elev_mean"]="Elevation (m)"
pp_DL$vars[pp_DL$vars=="frac_snow"]="Snow fraction"
pp_DL$vars[pp_DL$vars=="stream_elas"]="Streamflow elasticity"
#pp_DL$vars[pp_DL$vars=="pet_mean"]="Potential\nevapotranspiration (mm/day)"
pp_DL$vars[pp_DL$vars=="pet_mean"]=NA
pp_DL$vars[pp_DL$vars=="low_q_freq"]="Low flow\nfrequency (days/year)"
pp_DL$vars[pp_DL$vars=="q_mean"]="Streamflow (mm/day)"
pp_DL$vars[pp_DL$vars=="baseflow_index"]="Baseflow index"
pp_DL$vars[pp_DL$vars=="low_q_dur"]="Low flow\nDuration (days)"
pp_DL$vars[pp_DL$vars=="p_seasonality"]="Precipitation\nseasonality"
#pp_DL$vars[pp_DL$vars=="hfd_mean"]="Half flow\ndate (day of year)"
pp_DL$vars[pp_DL$vars=="p_mean"]="Precipitation (mm/day)"
pp_DL$vars[pp_DL$vars=="high_q_freq"]="High flow\nfrequency (days/year)"
pp_DL$vars[pp_DL$vars=="q95"]="Q95 (mm/day)"
#pp_DL$vars[pp_DL$vars=="soil_depth_pelletier"]="depth to bedrock (m)"
pp_DL$vars[pp_DL$vars=="soil_depth_pelletier"]=NA
pp_DL$vars[pp_DL$vars=="aridity"]=NA
pp_DL$vars[pp_DL$vars=="hfd_mean"]=NA

pp_DL$vars[pp_DL$vars=="high_prec_freq"]="High precipitation\nfrequency (days/year)"
orders = c("Snow fraction","Elevation (m)","Precipitation (mm/day)","Low flow\nDuration (days)",
           "Low flow\nfrequency (days/year)","Baseflow index")
pp_DL$`_vname_` = factor(pp_DL$vars,levels=orders)
drops <- c("vars","_ids_","_label_")

moddata=pp_DL[ , !(names(pp_DL) %in% drops)]
moddata = na.omit(moddata)
names(moddata)=c("type","x","y")
moddata$type <- factor(moddata$type, levels = orders)

b<-ggplot(data=moddata,aes(x=x,y=y))+geom_line()+
  facet_wrap(~type,scales = "free_x",nrow = 2)+theme_bw()+theme(panel.grid.major = element_blank(),
                                                                panel.grid.minor = element_blank(),text = element_text(size=15))+labs(y=expression("Average effect on"~Delta~"NSE"[Hybrid1-PB]),x="")

png("./fig3.png",width=10, height=6, units="in",res=600)
b
dev.off()

png("./figS2.png",width=10, height=16, units="in",res=600)
a
dev.off()











rm(list = ls())

hyd_mod <- read.table("./NSE_hyd.txt", quote="\"", comment.char="")
LSTM <- read.csv("./NSE_pure_datadriven_all.txt", sep="")
hybrid1 <- read.csv("./NSE_hybrid1_all.txt", sep="")
hybrid2 <- read.csv("./NSE_hybrid2_all.txt", sep="")
library('readr')
library('rgdal')
library(ggpubr)
library(gridExtra)

basin_list <- read_csv("./basin_list.txt", 
                       col_names = FALSE)
names(basin_list)=c("gauge_id")
DL_data <- cbind.data.frame(LSTM,hybrid1,hybrid2,hyd_mod)
DL_data[DL_data<0]=-0.01
names(DL_data)=c("LSTM","Hybrid1","Hybrid2","PB")
catchs = NA
catchs[1] = length(which(DL_data$PB<=0))
catchs[2] = length(which(DL_data$PB>0 & DL_data$PB<=(0.5)))
catchs[3] = length(which(DL_data$PB>0.5 & DL_data$PB<=(0.75)))
catchs[4] = length(which(DL_data$PB>0.75))
catchdata = cbind.data.frame(catchs,c("NSE<=0", "0<NSE<=0.5", "0.5<NSE<=0.75","NSE>0.75"),-0.5)
names(catchdata)=c("No","division","ys")
molten=reshape::melt(DL_data)
model_data <- rep(hyd_mod,3)
model_ful <- do.call(cbind.data.frame,model_data)
model_ful[model_ful<0]=-0.01
DL_data_diff <- cbind.data.frame((DL_data[,1:3] - model_ful),hyd_mod)

names(DL_data_diff )=c("LSTM","Hybrid1","Hybrid2","PB")
moltendata = reshape::melt(DL_data_diff,id.vars = "PB")

moltendata$division[moltendata$PB<0 |moltendata$PB==0]="NSE<=0"
moltendata$division[moltendata$PB>0 & moltendata$PB<(0.5)]="0<NSE<=0.5"
moltendata$division[moltendata$PB>=0.5 & moltendata$PB<(0.75)]="0.5<NSE<=0.75"
moltendata$division[moltendata$PB>=0.75 ]="NSE>0.75"
moltendata$division <- factor(moltendata$division,levels = c("NSE<=0", "0<NSE<=0.5", "0.5<NSE<=0.75","NSE>0.75"))
b<-ggplot() + geom_hline(yintercept=0, linetype="dotdash",color = "black")+
  geom_boxplot(data=moltendata,aes(x=division,y=value,color=variable,fill=variable),alpha=0.6)+theme_bw()+  scale_color_manual(values = c("#440154FF", "#38598CFF", "#C2DF23FF"))+
  scale_fill_manual(values = c("#440154FF", "#38598CFF", "#C2DF23FF"))+
  geom_text(data=catchdata,aes(x=division,y=ys,label = paste0("N = ",No)),size = 2.5)+
  theme_bw()+theme(legend.background=element_blank(),legend.position="bottom")+scale_x_discrete(labels = c(expression(NSE[PB]~"<=0"),expression("0<"~NSE[PB]~"<0.5"), expression("0.5<="~NSE[PB]~"<=0.75"), expression(NSE[PB]~">0.75")))+
  labs(x="PB Performance",y=expression(Delta~"NSE"["DL"]),color="",fill="",title="(a)")+coord_flip()



hyd_mod <- read.table("./NSE_hyd.txt", quote="\"", comment.char="")
LSTM <- read.csv("./NSE_pure_datadriven_all.txt", sep="")
hybrid1 <- read.csv("./NSE_hybrid1_all.txt", sep="")
hybrid2 <- read.csv("./NSE_hybrid2_all.txt", sep="")
library('readr')
library('rgdal')
library(ggpubr)

basin_list <- read_csv("./basin_list.txt", 
                       col_names = FALSE)
names(basin_list)=c("gauge_id")
DL_data <- cbind.data.frame(LSTM,hybrid1,hybrid2)
names(DL_data)=c("LSTM","Hybrid1","Hybrid2")
DL_data[DL_data<0]=-0.01

model_data <- rep(hyd_mod,3)
model_ful <- do.call(cbind.data.frame,model_data)
model_ful[model_ful<0]=-0.01

DL_data_diff <- (DL_data - model_ful)
ML_model <-cbind(basin_list,DL_data_diff)
camels_topo <- read_delim("./camels_attributes_v2.0/camels_topo.txt",  ";", escape_double = FALSE, trim_ws = TRUE)
camels_clim <- read_delim("./camels_attributes_v2.0/camels_clim.txt",  ";", escape_double = FALSE, trim_ws = TRUE)
camels_hydro <- read_delim("./camels_attributes_v2.0/camels_hydro.txt",  ";", escape_double = FALSE, trim_ws = TRUE)
camels_soil <- read_delim("./camels_attributes_v2.0/camels_soil.txt",  ";", escape_double = FALSE, trim_ws = TRUE)
camels_vege <- read_delim("./camels_attributes_v2.0/camels_vege.txt",  ";", escape_double = FALSE, trim_ws = TRUE)
#camels_data = cbind.data.frame(camels_topo[,1:5],camels_clim[,c(2,3,4,5,6,7,8,10,11)],camels_hydro[,2:14],camels_soil[,2:12],camels_vege[,c(2,3,4,5,6,7,9,10)])
camels_data = cbind.data.frame(camels_topo[,1:5],camels_clim[,c(2,3,4,5,6,7,8,10,11)],camels_hydro[,2:14])

us_states = readOGR(dsn="./USA_Shape",layer='states')
map_plot <-merge(ML_model,camels_data[,1:3],by="gauge_id")


molten = reshape::melt(map_plot[,2:6],id=c("gauge_lat","gauge_lon"))
library(ggplot2)
LtoM <-colorRampPalette(c('red', 'yellow' ))
MtoH <-colorRampPalette(c('lightgreen', 'darkgreen'))
a<-ggplot(na.omit(molten),aes(x=gauge_lon,y=gauge_lat,color=value),color="black") + 
  geom_polygon(data = us_states,aes(x=long, y=lat, group=group),fill="white",color="black")+
  geom_point(size=1)+
  facet_wrap(~variable)+scale_color_gradient2(low=LtoM(100), mid='snow3', 
                                             high=MtoH(100), space='Lab')+
  labs(color=expression(Delta~"NSE"[DL]),title = "(c)",x="lat",y="long")+theme_bw()+
  theme(text = element_text(size=15),legend.position = "bottom")+guides(color = guide_colourbar(barwidth= 20))
png("./figS1.png",width=10, height=4, units="in",res=600)
a
dev.off()
library(plyr)
mu <- ddply(molten, "variable", summarise, grp.mean=mean(value,na.rm = TRUE))
head(mu)
molten$variable <- factor(molten$variable, levels = rev(levels(molten$variable)))

c<-ggplot(molten,aes(x=value,color=variable,fill=variable))+geom_density(alpha=0.6)+  scale_color_manual(values =  c("#C2DF23FF", "#38598CFF","#440154FF"))+
  scale_fill_manual(values = c("#C2DF23FF", "#38598CFF","#440154FF"))+
  theme_bw()+theme(legend.background=element_blank(),text = element_text(size=20))+ geom_vline(aes(xintercept=0),linetype="dotdash", 
                                                                  color = "black")+
  labs(x=expression(Delta~"NSE"[DL]),color="",fill="",title="(b)")+theme(legend.position = "bottom")
my_layout = rbind(c(2,3),
                  c(2,3),
                  c(1,1),
                  c(1,1),
                  c(1,1))
png("./fig2.png",width=10, height=4, units="in",res=600)
ggarrange(b,c,widths = c(0.55,0.45),common.legend = TRUE,legend = "bottom")
dev.off()


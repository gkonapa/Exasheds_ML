rm(list = ls())
hyd_mod <- read.table("./NSE_hyd.txt", quote="\"", comment.char="")
LSTM <- read.csv("./NSE_pure_datadriven_all.txt", sep="")
hybrid1 <- read.csv("./NSE_hybrid1_all.txt", sep="")
hybrid2 <- read.csv("./NSE_hybrid2_all.txt", sep="")
library('readr')
library('rgdal')
library(ggpubr)
library(gridExtra)
library(ggcorrplot)
library(ggridges)

basin_list <- read_csv("./basin_list.txt", 
                       col_names = FALSE)
names(basin_list)=c("gauge_id")
DL_data <- cbind.data.frame(LSTM,hybrid1,hybrid2,hyd_mod)
DL_data[DL_data<0]=-0.01
names(DL_data)=c("LSTM","Hybrid1","Hybrid2","PB")
ML_model <-cbind(basin_list,DL_data)

molten=reshape::melt(DL_data)
camels_topo <- read_delim("./camels_attributes_v2.0/camels_topo.txt",  ";", escape_double = FALSE, trim_ws = TRUE)

us_states = readOGR(dsn="./USA_Shape",layer='states')
map_plot <-merge(ML_model,camels_topo[,1:3],by="gauge_id")
molten = reshape::melt(map_plot[,2:7],id=c("gauge_lat","gauge_lon"))
molten=na.omit(molten)
molten$variable <- factor(molten$variable,levels = c("PB","LSTM","Hybrid1","Hybrid2"))

library(ggplot2)
a<-ggplot(molten,aes(x=gauge_lon,y=gauge_lat,color=value),color="black") + 
  geom_polygon(data = us_states,aes(x=long, y=lat, group=group),fill="white",color="black")+
  geom_point(size=1.5)+
  facet_wrap(~variable)+scale_colour_distiller(palette = "RdYlGn",direction = 1)+
  labs(color="NSE",title = "(a)",x="lon",y="lat")+theme_bw()+
  theme(legend.position = "bottom",text = element_text(size=15))+
  guides(color = guide_colourbar(barwidth= 20))

b<- ggplot(molten,aes(value,color=variable)) + 
  stat_ecdf(geom = "step")+theme_bw()+
  labs(x="NSE",y="CDF",color="",title="(b)")+
  scale_color_manual(values = c("red","#440154FF", "#38598CFF", "#C2DF23FF"))+
  theme(legend.background=element_blank())+
  guides(color=guide_legend(nrow=4,byrow=TRUE))

corr <- round(cor(na.omit(DL_data)), 2)
p.mat <- cor_pmat(na.omit(DL_data))
c<-ggcorrplot(corr, hc.order = TRUE, type = "lower",
           outline.col = "white",
           ggtheme = ggplot2::theme_bw,lab = TRUE,
           colors = c("red", "yellow", "#007600"))+labs(title = "(c)")

my_layout = rbind(c(1,1,1,2,2),
                  c(1,1,1,3,3))
png("./fig1.png",width=14, height=8, units="in",res=600)
grid.arrange(a,b,c,layout_matrix=my_layout)
dev.off()


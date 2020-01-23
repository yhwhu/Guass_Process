a=flights[,4]
b=flights[,8]
c=flights[,23]
flights2<-as.data.frame(cbind(a,b,c))
names(airports)[1]<-c("ORIGIN_AIRPORT")
a1=airports[,1]
a2=airports[,6]
a3=airports[,7]
airports2<-as.data.frame(cbind(a1,a2,a3))
merge_data=merge(flights2,airports2,by='ORIGIN_AIRPORT')
list1<-unique(flights2$ORIGIN_AIRPORT)
list2<-unique(merge_data$ORIGIN_AIRPORT)
list3<-unique(merge_data$port_day)
###去掉存在缺失值的行
merge_data<-na.omit(merge_data)
###按组计算均值
merge_data$port_day=paste(merge_data$ORIGIN_AIRPORT,merge_data$DAY_OF_WEEK)
list<-unique(merge_data$port_day)
mean_delay<-tapply(merge_data$ARRIVAL_DELAY,merge_data$port_day,mean)
####
mean_data<-merge_data[!duplicated(merge_data$port_day),]
mean_data<-as.data.frame(cbind(mean_data,mean_delay))
mean_data=mean_data[,-6]
mean_data=mean_data[,-3]
mean_data$delay=mean_data$mean_delay
mean_data$delay[mean_delay>10]=1
mean_data$delay[mean_delay<10]=-1
write.csv(mean_data,"mean_data.csv")


## ep
Rcpp::sourceCpp('ep_openmp.cpp')
library(MASS)
library(kernlab)
library(ggplot2)
library(gridExtra)
library(plotly)

df <- read.csv("data/mean_data.csv")
df <- df[,-(1:2)]
X <- as.matrix(df[,2:4])
y <- df$delay

model_without_select = function(train.x, train.y, test.x, theta){
  K <- kernal_compute(train.x, train.x, theta, 1)
  res <-  ep_train(K, train.y)
  v <-  res[[1]]
  tau <-  res[[2]]
  res_predict <- ep_predict(K, v, tau, train.x, theta, 1, test.x)
  return(res_predict)
}

# train by all data
pred1 <-   model_without_select(X, y, X, 1)
pred1[pred1 > 0.5] <- 1
pred1[pred1 <= 0.5] <- -1
sum(pred1==y) / 2223

# train set and test set
N <- nrow(df)
test.size <- 1000
index <- sample(1:N, test.size)
train.x <- X[-index,,drop=FALSE]
train.y <- y[-index,drop=FALSE]
test.x <- X[index,,drop=FALSE]
test.y <- y[index,drop=FALSE]
pred2 <-   model_without_select(train.x, train.y, test.x, 1)
pred2[pred2 > 0.5] <- 1
pred2[pred2 <= 0.5] <- -1
sum(pred2==test.y) / test.size

# usa
X <- as.data.frame(X)
y <- as.matrix(y[X$LONGITUDE > -130 & X$LATITUDE < 60 & X$LATITUDE > 20])
X <- as.matrix(X[X$LONGITUDE > -130 & X$LATITUDE < 60 & X$LATITUDE > 20,])
df_expand <- expand.grid(lat = seq(min(X[,2]) - 5, max(X[,2]) + 5, length.out = 10),
                       long = seq(min(X[,3]) - 5, max(X[,3]) + 5, length.out = 10))
K <- kernal_compute(X, X, 1, 1)
res <-  ep_train(K, y)
v <-  res[[1]]
tau <-  res[[2]]

for (i in 1:7) {
  test.x <- as.matrix(cbind(day_of_week = i, df_expand))
  res_predict <- ep_predict(K, v, tau, X, 1, 1, test.x)
  df_plot <- cbind(df_expand, res_predict)
  
  # jpeg(paste("day_of_week" ,i ,".jpeg",sep=""), width = 600, height = 400, quality = 100)
  p <- ggplot(df_plot, aes(long, lat, fill= res_predict)) + 
    geom_tile()+ scale_fill_gradient(low="green", high="red") +
    ggtitle(paste("day_of_week",i,sep=""))
  p
  # ggplotly(p, tooltip="text")
  # dev.off()
}



# states_map <- map_data("state")
# ggplot(states_map, aes(x=long,y=lat,group=group)) +
#   geom_polygon(fill="white",colour="black") +
#   labs(title = "USA Map")
# 
# head(USArrests)
# crimes <- data.frame(state= tolower(rownames(USArrests)), USArrests)
# crime_map <- merge(states_map,crimes,by.x="region",by.y = "state")
# head(crime_map)
# ggplot(crime_map, aes(x=long,y=lat, group = group, fill = Assault)) +
#   geom_polygon(colour = "black") +
#   coord_map("polyconic") +
#   labs(title = "USA Map")

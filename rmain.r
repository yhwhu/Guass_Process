Rcpp::sourceCpp('ep_openmp.cpp')
library(MASS)
library(kernlab)
library(ggplot2)
library(gridExtra)

model_with_select = function(train.x, train.y, test.x, theta){
  theta <- gradient_descent(train.x, train.y, theta, 1)
  K <- kernal_compute(train.x, train.x, theta, 1)
  res <-  ep_train(K, train.y)
  v <-  res[[1]]
  tau <-  res[[2]]
  res_predict <- ep_predict(K, v, tau, train.x, theta, 1, test.x)
  return(res_predict)
}


model_without_select = function(train.x, train.y, test.x, theta){
  K <- kernal_compute(train.x, train.x, theta, 1)
  res <-  ep_train(K, train.y)
  v <-  res[[1]]
  tau <-  res[[2]]
  res_predict <- ep_predict(K, v, tau, train.x, theta, 1, test.x)
  return(res_predict)
}


accuracy <- function(pred1, pred2, test.y){
  test.size <- length(test.y)
  accuracy1 <- rep(-1, test.size)
  accuracy1[pred1 > 0.5] <- 1
  accu1 <- sum(accuracy1 == test.y)/test.size
  
  accuracy2 <- rep(-1, test.size)
  accuracy2[pred2 > 0.5] <- 1
  accu2 <- sum(accuracy2 == test.y)/test.size
  return(c(accu1 = accu1, accu2 = accu2))
}

set.seed(20)
N <- 1000
p <- 2
sigma <- matrix(c(1, 0.25, 0.25, 1), nrow = 2)
data.x1 <- mvrnorm(n = N/2, mu = rep(-1,p), Sigma = sigma)
data.x2 <- mvrnorm(n = N/2, mu = rep(1,p), Sigma = sigma)
data.x <- rbind(data.x1, data.x2)
data.y <- c(rep(-1, N/2), rep(1, N/2))
test.size <- 100
index <- sample(1:N, test.size)
train.x <- data.x[-index,,drop=FALSE]
train.y <- data.y[-index,drop=FALSE]
test.x <- data.x[index,,drop=FALSE]
test.y <- data.y[index,drop=FALSE]

pred1 <-   model_with_select(train.x, train.y, test.x, theta=1)
pred2 <-   model_without_select(train.x, train.y, test.x, 1)
pred2[pred2 > 0.5] <- 1
pred2[pred2 < 0.5] <- -1
sum(pred2==test.y) / test.size


accuracy(pred1, pred2, test.y)


res <- microbenchmark::microbenchmark(
                               # pred1 =  model_with_select(train.x, train.y, test.x, 10),
                               pred2 =  model_without_select(train.x, train.y, test.x, 1),
                               times = 5)

# plot all data
data_sample <- data.frame(x1=data.x[,1],x2=data.x[,2],y=as.character(data.y))
p_sample <- ggplot(data_sample) + geom_point(aes(x1,x2,color=y)) 
p_sample

# plot predict for selecting
train_p <- cbind(train.x, train.y)
train_p <- data.frame(train_p)
p1 <- ggplot(train_p, aes(x=V1, y=V2, color=train.y)) + 
  geom_point(size=3) + ggtitle("Train data in two classes")

test_p <- cbind(test.x, pred1, test.y)
test_p <- data.frame(test_p)
p2 <- ggplot(test_p, aes(x=V1, y=V2, color=test.y)) + 
  geom_point(size=3) + ggtitle("Test data in two classes")

p3 <- ggplot(test_p, aes(x=V1, y=V2, color=pred1)) + 
  geom_point(size=3) + ggtitle("Test data prediction of prob")

# plot predict for not selecting
p4 <- ggplot(test_p, aes(x=V1, y=V2, color=pred2)) + 
  geom_point(size=3) + ggtitle("Test data prediction of prob")

l <- list(p1, p2, p3, p4)
grid.arrange(grobs = l, ncol = 4)


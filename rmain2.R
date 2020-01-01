Rcpp::sourceCpp('ep_op2.cpp')
library(MASS)
library(kernlab)
library(ggplot2)
library(gridExtra)

model = function(train.x, train.y, test.x){
  K <- kernal_compute_paralle(train.x, train.x, 1)
  res <-  ep_train(K, train.y)
  v <-  res[[1]]
  tau <-  res[[2]]
  res_predict <- ep_predict_paralle_vec(K, v, tau, train.x, 1, test.x)
  return(res_predict)
}


set.seed(20)
N <- 100
p <- 2
sigma <- matrix(c(1, 0.25, 0.25, 1), nrow = 2)
data.x1 <- mvrnorm(n = N/2, mu = rep(-1,p), Sigma = sigma)
data.x2 <- mvrnorm(n = N/2, mu = rep(1,p), Sigma = sigma)
data.x <- rbind(data.x1, data.x2)
data.y <- c(rep(-1, N/2), rep(1, N/2))
test.size <- 10
index <- sample(1:N, test.size)
train.x <- data.x[-index,,drop=FALSE]
train.y <- data.y[-index,drop=FALSE]
test.x <- data.x[index,,drop=FALSE]
test.y <- data.y[index,drop=FALSE]

pred4 <-  model(train.x, train.y, test.x)
pred4

pred5 <-  model_paralle_vec2(train.x, train.y, test.x)
pred5
sum(pred4 - pred5)

res <- microbenchmark::microbenchmark(
  pred4 =  model(train.x, train.y, test.x),
  # pred5 =  model_paralle_vec2(train.x, train.y, test.x),
  times = 3)
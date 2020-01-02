
Rcpp::sourceCpp('ep_openmp.cpp')
library(MASS)
library(kernlab)
library(ggplot2)
library(gridExtra)

model = function(train.x, train.y, test.x, theta){
  theta <- gradient_descent(train.x, train.y, theta, 1)
  K <- kernal_compute_paralle(train.x, train.x, theta, 1)
  res <-  ep_train(K, train.y)
  v <-  res[[1]]
  tau <-  res[[2]]
  res_predict <- ep_predict_paralle_vec(K, v, tau, train.x, theta, 1, test.x)
  return(res_predict)
}

model2 = function(train.x, train.y, test.x, theta){
  K <- kernal_compute_paralle(train.x, train.x, theta, 1)
  res <-  ep_train(K, train.y)
  v <-  res[[1]]
  tau <-  res[[2]]
  # theta <- gradient_descent(train.x, train.y, theta, 1)
  res_predict <- ep_predict_paralle_vec(K, v, tau, train.x, theta, 1, test.x)
  return(res_predict)
}

set.seed(20)
N <- 100
p <- 2
sigma <- matrix(c(1, 0.25, 0.25, 1), nrow = 2)
data.x1 <- mvrnorm(n = N/2, mu = rep(-2,p), Sigma = sigma)
data.x2 <- mvrnorm(n = N/2, mu = rep(2,p), Sigma = sigma)
data.x <- rbind(data.x1, data.x2)
data.y <- c(rep(-1, N/2), rep(1, N/2))
test.size <- 10
index <- sample(1:N, test.size)
train.x <- data.x[-index,,drop=FALSE]
train.y <- data.y[-index,drop=FALSE]
test.x <- data.x[index,,drop=FALSE]
test.y <- data.y[index,drop=FALSE]

pred1 =  model(train.x, train.y, test.x, 1)
pred2 =  model2(train.x, train.y, test.x, -68.2569)

res <- microbenchmark::microbenchmark(
                               pred1 =  model(train.x, train.y, test.x, 1),
                               pred2 =  model2(train.x, train.y, test.x, -68.2569),
                               times = 3)

# plot all data
data_sample <- data.frame(x1=data.x[,1],x2=data.x[,2],y=as.character(data.y))
p_sample <- ggplot(data_sample) + geom_point(aes(x1,x2,color=y)) 
p_sample

# plot predict
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

l <- list(p1, p2, p3)
grid.arrange(grobs = l, ncol = 3)


# test code
model_paralle_vec = function(train.x, train.y, test.x){
  K <- kernal_compute_paralle(train.x, train.x, 1)
  res <-  ep_train_paralle(K, train.y)
  v <-  res[[1]]
  tau <-  res[[2]]
  res_predict <- ep_predict_paralle_vec(K, v, tau, train.x, train.y, 1, test.x)
  return(res_predict)
}

model_paralle_mat = function(train.x, train.y, test.x){
  K <- kernal_compute_paralle(train.x, train.x, 1)
  res <-  ep_train_paralle(K, train.y)
  v <-  res[[1]]
  tau <-  res[[2]]
  res_predict <- ep_predict_paralle_mat(K, v, tau, train.x, train.y, 1, test.x)
  return(res_predict)
}

res <- microbenchmark::microbenchmark(pred1 = model(train.x, train.y, test.x), 
                                      pred2 = model_paralle(train.x, train.y, test.x), 
                                      times = 1)

microbenchmark::microbenchmark(res1 = kernal_compute(train.x, train.x, 1), 
                               res2 = kernal_compute_paralle(train.x, train.x, 1), 
                               times = 10)

microbenchmark::microbenchmark(res1 = computeCov(t(train.x)),
                               res2 = computeCov_parallel(t(train.x)),
                               times=10)

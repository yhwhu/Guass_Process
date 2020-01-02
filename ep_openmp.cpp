#include <RcppArmadillo.h>
#include <vector>
#include <omp.h>
#include <chrono>
using namespace arma;
using namespace std;

// [[Rcpp::plugins(openmp)]]
// [[Rcpp::depends(RcppArmadillo)]]

field<vec> ep_train(const mat & X, const vec & y);

vec ep_predict(const mat & K, const vec & v, const vec & tau, const mat & X, vec theta, int kernal_type, const mat &  x_test);

vec gradient_descent(const mat & X, const vec & y, vec theta, int kernal_type);
vec ep_model_select(const mat & X, const vec & y, vec theta, int kernal_type);

mat kernal_compute(const mat & X1, const mat & X2, vec theta, int kernal_type);
inline double kernal_func(const rowvec & x1, const rowvec & x2, vec theta, int kernal_type);


// [[Rcpp::export]]
field<vec> ep_train(const mat & K, const vec & y) {
  auto t1 = chrono::high_resolution_clock::now();
  
  int n = y.n_elem;
  cout << "Train sample size is: " << n;
  vec v =  zeros<vec>(n), tau = v, mu = v;
  mat sigma = K;
  int iter = 0, max_iter = 20;
  double tau_i, v_i, tau_i_1, z_i, N_z, Phi_z, mu_hat, sigma_hat, del_tau;
  while( iter++ < max_iter){
    vec tmp = v;
    for(int i = 0; i < n; i++){
      tau_i = 1 / sigma(i, i) - tau(i);
      v_i = 1 / sigma(i, i) * mu(i) - v(i);
      
      tau_i_1 = 1 / tau_i;
      z_i = y(i) * v_i * tau_i_1 / sqrt(1 + tau_i_1);
      N_z = normpdf(z_i), Phi_z = normcdf(z_i);
      
      mu_hat = v_i / tau_i + y(i)* tau_i_1 * N_z / (Phi_z * sqrt(1 + tau_i_1));
      sigma_hat = tau_i_1 - pow(tau_i_1, 2) * N_z  /
        ((1 + tau_i_1) * Phi_z) * (z_i + N_z / Phi_z);
      
      del_tau = 1 / sigma_hat - tau_i - tau(i);
      tau(i) += del_tau;
      v(i) = 1 / sigma_hat * mu_hat - v_i;
      sigma -= 1 / (1 / del_tau + sigma(i,i)) * sigma.col(i) * (sigma.col(i).t());
      mu = sigma * v;
    }
    if(norm(v-tmp, 2) < 1e-3){
      cout << ". Train iter: " << iter << ".\n";
      break;
    }
    // mat S_2 = diagmat(sqrt(tau)), I_n = eye(n, n);
    // mat B = I_n + S_2 * K * S_2;
    // mat L = chol(B, "lower");
    // mat V = solve(L, I_n) * S_2 * K;
    // sigma =  K - V.t() * V;
    // mu = sigma * v;
  }
  field<vec> res = {v, tau};
  
  auto t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, std::milli> fp_ms = t2 - t1;
  cout << "The elapsed time to train: " << fp_ms.count()<< " milliseconds.\n";
  
  return res;
}

// [[Rcpp::export]]
vec ep_predict(const mat & K, const vec & v, const vec & tau, const mat & X, vec theta, int kernal_type, const mat &  x_test){
  auto t1 = chrono::high_resolution_clock::now();

  int n = K.n_rows;
  mat S_2 = diagmat(sqrt(tau)), I_n = eye(n, n);
  mat B = I_n + S_2 * K * S_2;
  mat L = chol(B, "lower");
  mat z1 = solve(trimatl(L), S_2*(K*v));
  mat z2 = solve(trimatu(L.t()), z1);
  vec z = S_2 * z2;
  
  int m = x_test.n_rows;
  vec f_test(m), Var(m), pai(m);
#pragma omp parallel for schedule(static)
  for(int i=0; i<m; i++){
    mat kx = kernal_compute(X, x_test.row(i), theta, kernal_type);
    f_test(i) = as_scalar(kx.t() * (v - z));
    vec v2 = solve(trimatl(L), S_2 * kx);
    Var(i) = as_scalar(kernal_compute(x_test.row(i), x_test.row(i), theta, kernal_type) - v2.t() * v2);
    pai(i) = normcdf(f_test(i) / sqrt(1 + Var(i)));
    // printf("Predict %dth samples, pai is: %f.\n", i, pai(i));
  }
  
  auto t8 = chrono::high_resolution_clock::now();
  chrono::duration<double, std::milli> fp_ms2 = t8 - t1;
  // chrono::duration<double, std::milli> fp_ms = t3 - t2, fp_ms2 = t8 - t1, fp_ms3 = t5 - t4;
  // cout << "(Paralle inner) The elapsed time to chol " << fp_ms3.count()<< " milliseconds.\n";
  // cout << "(Paralle inner) The elapsed time to solve and mul " << fp_ms.count()<< " milliseconds.\n";
  cout << "The elapsed time to predict: " << fp_ms2.count()<< " milliseconds.\n";

  return pai;
}

// [[Rcpp::export]]
vec gradient_descent(const mat & X, const vec & y, vec theta, int kernal_type){
  double alpha = 0.001;
  int max_iter = 30, iter = 0;
  vec grad;
  while(iter++ < max_iter){
    cout << "----Gradient descent iter: " << iter << ". theta is:" << theta;
    grad = ep_model_select(X, y, theta, kernal_type);
    cout << "Grad is:" << grad << endl;
    theta -= log(abs(grad)) * alpha * grad;
    if(norm(grad, 2) < 4){
      cout << "----Finish gradient descent, iter: " << iter << ". theta is:" << theta<<endl;
      break;
    }
    // alpha *= 0.9;
  }
  return theta;
}

vec ep_model_select(const mat & X, const vec & y, vec theta, int kernal_type){
  int dim_theta = theta.n_elem, n = X.n_rows;
  mat K = kernal_compute(X, X, theta, kernal_type);
  field<vec> train_res = ep_train(K, y);
  vec v = train_res(0), tau = train_res(1);
  mat S_2 = diagmat(sqrt(tau)), I_n = eye(n, n);
  mat B = I_n + S_2 * K * S_2;
  mat L = chol(B, "lower");
  mat b1 = solve(trimatu(L.t()), S_2*(K*v));
  mat b2 = solve(trimatl(L), b1);
  vec b = v - S_2 * b2;
  mat R1 = solve(trimatl(L), S_2);
  mat R2 = solve(trimatu(L.t()), R1);
  mat R = b * b.t() - S_2 * R2;
  
  mat C;
  vec grad = zeros<vec>(dim_theta);
  for(int j = 0; j < dim_theta; j++){
    switch (kernal_type){
    case 1:
      C = kernal_compute(X, X, theta, 11);
    }
    // cout << "sum C: " << accu(C) <<  "   sum Rt"<< accu(R.t()) << endl;
    grad(j) = accu(C % R.t());
  }
  grad /= 2;
  return grad;
}

// [[Rcpp::export]]
mat kernal_compute(const mat & X1, const mat & X2, vec theta, int kernal_type){
  
  // auto t1 = chrono::high_resolution_clock::now();
  
  int n = X1.n_rows, m = X2.n_rows;
  mat res(n, m);
#pragma omp parallel for schedule(static) collapse(2)
  for(int i = 0; i < n; i++){
    for(int j = 0; j < m; j++){
      res(i, j) = kernal_func(X1.row(i), X2.row(j), theta, kernal_type);
    }
  }
  
  // auto t2 = chrono::high_resolution_clock::now();
  // chrono::duration<double, std::milli> fp_ms = t2 - t1;
  // cout << "(Paralle) X1 row is: " << n <<". X2 row is " << m;
  // cout << "The elapsed time to compute K: " << fp_ms.count()<< " milliseconds.\n";
  // res = symmatl(res);
  return res;
}

inline double kernal_func(const rowvec & x1, const rowvec & x2, vec theta, int kernal_type){
  double res, theta1 = theta(0);
  switch (kernal_type){
  case 1:
    res = norm(x1-x2);
    res = exp(- res * res / (2 * theta1 * theta1));
    break;
  case 11:
    res = norm(x1-x2);
    res = res * res * exp(- res * res / (2 * theta1 * theta1)) / (pow(theta1, 3));
    break;
  }
  return res;
}


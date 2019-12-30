# include <RcppArmadillo.h>
#include <trng/normal_dist.hpp>
#include <vector>
#include <chrono>


using namespace arma;
using namespace std;
// [[Rcpp::depends(RcppArmadillo)]]

vector<vec> ep_train(const mat & X, const vec & y);
double ep_predict(const mat & K, vec v, vec tau, mat X, vec y, int kernal_type, mat x_test);
mat kernal_compute(const mat & X1, const mat & X2, int kernal_type);
inline double kernal_func(const rowvec & x1, const rowvec & x2, int kernal_type);

// [[Rcpp::export]]
vector<vec> ep_train(const mat & K, const vec & y) {
  auto t1 = chrono::high_resolution_clock::now();
  
  trng::normal_dist<> u(0, 1);
  int n = y.n_elem;
  vec v =  zeros<vec>(n), tau = v, mu = v;
  mat sigma = K;
  int iter = 0, max_iter = 50;
  double tau_i, v_i, tau_i_1, z_i, N_z, Phi_z, mu_hat, sigma_hat, del_tau;
  while( iter++ < max_iter){
    vec tmp = v;
    for(int i = 0; i < n; i++){
      tau_i = 1 / sigma(i, i) - tau(i);
      v_i = 1 / sigma(i, i) * mu(i) - v(i);
      
      tau_i_1 = 1 / tau_i;
      z_i = y(i) * v_i * tau_i_1 / sqrt(1 + tau_i_1);
      N_z = u.pdf(z_i), Phi_z = u.cdf(z_i);
      
      mu_hat = v_i / tau_i + y(i)* tau_i_1 * N_z / (Phi_z * sqrt(1 + tau_i_1));
      sigma_hat = tau_i_1 - pow(tau_i_1, 2) * N_z  /
        ((1 + tau_i_1) * Phi_z) * (z_i + N_z / Phi_z);
      
      del_tau = 1 / sigma_hat - tau_i - tau(i);
      tau(i) += del_tau;
      v(i) = 1 / sigma_hat * mu_hat - v_i;
      sigma -= 1 / (1 / del_tau + sigma(i,i)) * sigma.col(i) * (sigma.col(i).t());
      mu = sigma * v;
    }
    if(norm(v-tmp, 2) < 1e-3) break;
    // mat S_2 = diagmat(sqrt(tau)), I_n = eye(n, n);
    // mat B = I_n + S_2 * K * S_2;
    // mat L = chol(B, "lower");
    // mat V = solve(L, I_n) * S_2 * K;
    // sigma =  K - V.t() * V;
    // mu = sigma * v;
  }
  vector<vec> res(2);
  res.at(0) = v;
  res.at(1) = tau;
  
  auto t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, std::milli> fp_ms = t2 - t1;
  cout << "(iter minus)The elapsed time to train " << fp_ms.count()<< " milliseconds.\n";
  
  return res;
}

// [[Rcpp::export]]
double ep_predict(const  mat & K, vec v, vec tau, mat X, vec y, int kernal_type, mat x_test){
  auto t1 = chrono::high_resolution_clock::now();
  
  int n = X.n_rows;
  mat kx = kernal_compute(X, x_test, kernal_type);

  mat S_2 = diagmat(sqrt(tau)), I_n = eye(n, n);
  mat B = I_n + S_2 * K * S_2;
  mat L = chol(B, "lower");
  mat L_1 = solve(L, I_n), Lt_1 = solve(L.t(), I_n);
  
  vec z = S_2 * Lt_1 * L_1 * S_2 * K * v;
  
  double f_test = as_scalar(kx.t() * (v - z));
  
  vec v2 = L_1 * S_2 * kx;
  double Var = as_scalar(kernal_compute(x_test, x_test, kernal_type) - v2.t() * v2);
  
  trng::normal_dist<> u(0, 1);
  double pai = u.cdf(f_test / sqrt(1 + Var));
  
  auto t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, std::milli> fp_ms = t2 - t1;
  cout << "(Not paralle) The elapsed time to predict " << fp_ms.count()<< " milliseconds.\n";
  
  return pai;
}

// [[Rcpp::export]]
mat kernal_compute(const mat & X1, const mat & X2, int kernal_type){
  auto t1 = chrono::high_resolution_clock::now();
  
  int n = X1.n_rows, m = X2.n_rows;
  mat res(n, m);
 
  for(int i = 0; i < n; i++){
    for(int j = 0; j < m; j++){
      res(i, j) = kernal_func(X1.row(i), X2.row(j), kernal_type);
    }
  }
  
  auto t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, std::milli> fp_ms = t2 - t1;
  cout << "(Not paralle) X1 row is: " << n <<". X2 row is " << m;
  cout << ". The elapsed time to compute K " << fp_ms.count()<< " milliseconds.\n";
  
  return res;
}

inline double kernal_func(const rowvec & x1, const rowvec & x2, int kernal_type){
  double res;
  // switch (kernal_type){
  // case 1:
  //   double sigma = 0.05;
  //   res = exp(-pow(norm(x1-x2, 2), 2)/sigma);
  // }
  res = norm(x1-x2);
  res = exp(- res * res / 0.05);
  return res;
}
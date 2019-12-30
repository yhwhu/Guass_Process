# include <RcppArmadillo.h>
#include <vector>
#include <omp.h>
#include <chrono>
using namespace arma;
using namespace std;

// [[Rcpp::plugins(openmp)]]
// [[Rcpp::depends(RcppArmadillo)]]

field<vec> ep_train(const mat & X, const vec & y);
// double ep_predict_paralle(const mat & K, vec v, vec tau, mat X, vec y, int kernal_type, rowvec x_test);
// vec ep_predict_paralle_vec(const mat & K, vec v, vec tau, mat X, vec y, int kernal_type, mat x_test);
mat kernal_compute_paralle(const mat & X1, const mat & X2, int kernal_type);
// vec kernal_compute_diag(const mat & X1, int kernal_type);
inline double kernal_func_paralle(const rowvec & x1, const rowvec & x2, int kernal_type);
// vec ep_predict_paralle_mat(const  mat & K, vec v, vec tau, mat X, vec y, int kernal_type, mat x_test);
vec ep_predict_paralle_vec(const mat & S_2, const mat & L, const vec & v, const vec &  z, const mat &  X, const vec &  y, int kernal_type, const mat &  x_test);
    
vec ep_model_select(const mat & L, const vec & b, const mat & R);
field<mat> compute_vec(const mat & K, const vec & v, const vec & tau);
// field<mat> compute_vec2(const mat & K, const vec & v, const vec & tau);

// [[Rcpp::export]]
field<vec> ep_train(const mat & K, const vec & y) {
  auto t1 = chrono::high_resolution_clock::now();
  
  int n = y.n_elem;
  cout << "Train sample size is: " << n;
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
field<mat> compute_vec(const mat & K, const vec & v, const vec & tau){
  auto t1 = chrono::high_resolution_clock::now();
  // predict
  int n = K.n_rows;
  mat S_2 = diagmat(sqrt(tau)), I_n = eye(n, n);
  mat B = I_n + S_2 * K * S_2;
  mat L = chol(B, "lower");
  mat z1 = solve(trimatl(L), S_2*K*v);
  mat z2 = solve(trimatu(L.t()), z1);
  vec z = S_2 * z2;
  
  
  // model select
  mat b1 = solve(trimatu(L.t()), S_2*K*v);
  mat b2 = solve(trimatl(L), b1);
  vec b = v - S_2 * z2;
  
  mat R1 = solve(trimatl(L), S_2);
  mat R2 = solve(trimatu(L.t()), R1);
  mat R = b * b.t() - S_2 * R2;
  
  field<mat> res = {S_2, L, z, b, R};
  auto t2 = chrono::high_resolution_clock::now();
  chrono::duration<double, std::milli> fp_ms2 = t2 - t1;
  cout << "The elapsed time to compute vec: " << fp_ms2.count()<< " milliseconds.\n";
  return res;
}


// [[Rcpp::export]]
vec ep_predict_paralle_vec(const mat & S_2, const mat & L, const vec & v, const vec &  z, const mat &  X, const vec &  y, int kernal_type, const mat &  x_test){
  auto t1 = chrono::high_resolution_clock::now();
  
  int m = x_test.n_rows;
  vec f_test(m), Var(m), pai(m);
#pragma omp parallel for schedule(static)
  for(int i=0; i<m; i++){
    mat kx = kernal_compute_paralle(X, x_test.row(i), kernal_type);
    f_test(i) = as_scalar(kx.t() * (v - z));
    vec v2 = solve(trimatl(L), S_2 * kx);
    Var(i) = as_scalar(kernal_compute_paralle(x_test.row(i), x_test.row(i), kernal_type) - v2.t() * v2);
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
vec ep_model_select(const mat & L, const vec & b, const mat & R, vec theta, int kernal_type){
  int dim_theta = theta.n_elem;
  double alpha;
  vec grad = zeros<vec>(dim_theta);
  mat C;
  switch (kernal_type){
  case 1:
    for(int j = 0; j < dim_theta; j++){
      for(int i = 0; i < R.n_rows; i++){
        grad(j) += as_scalar(R.row(i) * C.col(i));
      }
      grad(j) /= 2;
      theta(j) -= alpha * grad(j);
    }
  }
  return grad;
}

// [[Rcpp::export]]
mat kernal_compute_paralle(const mat & X1, const mat & X2, int kernal_type){
  // auto t1 = chrono::high_resolution_clock::now();
  
  int n = X1.n_rows, m = X2.n_rows;
  mat res(n, m);
#pragma omp parallel for schedule(static) collapse(2)
  for(int i = 0; i < n; i++){
    for(int j = 0; j < m; j++){
      res(i, j) = kernal_func_paralle(X1.row(i), X2.row(j), kernal_type);
    }
  }
  
  // auto t2 = chrono::high_resolution_clock::now();
  // chrono::duration<double, std::milli> fp_ms = t2 - t1;
  // cout << "(Paralle) X1 row is: " << n <<". X2 row is " << m;
  // cout << "The elapsed time to compute K: " << fp_ms.count()<< " milliseconds.\n";
  
  return res;
}

inline double kernal_func_paralle(const rowvec & x1, const rowvec & x2, int kernal_type){
  double res;
  // switch (kernal_type){
  // case 1:
  //   double sigma = 0.05;
  //   res = exp(-pow(norm(x1-x2, 2), 2)/sigma);
  // }
  double l = 0.05;
  res = norm(x1-x2);
  res = exp(- res * res / (2 * l * l));
  return res;
}



// // [[Rcpp::export]]
// field<mat> compute_vec2(const mat & K, const vec & v, const vec & tau){
//   auto t1 = chrono::high_resolution_clock::now();
//   // predict
//   int n = K.n_rows;
//   mat S_2 = diagmat(sqrt(tau)), I_n = eye(n, n);
//   mat B = I_n + S_2 * K * S_2;
//   mat L = chol(B, "lower");
//   vec tmp = S_2*K*v;
//   mat L_1 = solve(trimatl(L), I_n);
//   vec z = S_2 * (L_1.t() * (L_1 * tmp));
//   
//   
//   // model select
//   vec b = v - S_2 * (L_1 * (L_1.t() * tmp));
//   mat R = b * b.t() - S_2 * L_1.t() * L_1 * S_2;
//   
//   field<mat> res = {S_2, L, z, b, R};
//   auto t2 = chrono::high_resolution_clock::now();
//   chrono::duration<double, std::milli> fp_ms2 = t2 - t1;
//   cout << "The elapsed time to compute vec2: " << fp_ms2.count()<< " milliseconds.\n";
//   return res;
// }

// double ep_predict_paralle(const  mat & K, vec v, vec tau, mat X, vec y, int kernal_type, rowvec x_test){
//   auto t1 = chrono::high_resolution_clock::now();
// 
//   int n = X.n_rows;
//   mat kx = kernal_compute_paralle(X, x_test, kernal_type);
// 
//   mat S_2 = diagmat(sqrt(tau)), I_n = eye(n, n);
//   mat B = I_n + S_2 * K * S_2;
//   mat L = chol(B, "lower");
//   mat L_1 = solve(L, I_n), Lt_1 = solve(L.t(), I_n);
// 
//   vec z = S_2 * Lt_1 * L_1 * S_2 * K * v;
// 
//   double f_test = as_scalar(kx.t() * (v - z));
// 
//   vec v2 = L_1 * S_2 * kx;
//   double Var = as_scalar(kernal_compute_paralle(x_test, x_test, kernal_type) - v2.t() * v2);
// 
//   double pai = normcdf(f_test / sqrt(1 + Var));
// 
//   auto t2 = chrono::high_resolution_clock::now();
//   chrono::duration<double, std::milli> fp_ms = t2 - t1;
//   cout << "(Paralle) The elapsed time to predict " << fp_ms.count()<< " milliseconds.\n";
// 
//   return pai;
// }


// // [[Rcpp::export]]
// vec ep_predict_paralle_vec(const mat & K, vec v, vec tau, mat X, vec y, int kernal_type, mat x_test){
//   int n_test = x_test.n_rows;
//   vec pai_vec(n_test);
// #pragma omp parallel for schedule(static)
//   for(int i = 0; i<n_test; i++){
//     pai_vec(i) = ep_predict_paralle(K, v, tau, X, y, kernal_type, x_test.row(i));
//   }
//   return pai_vec;
// }

// // [[Rcpp::export]]
// vec ep_predict_paralle_mat(const  mat & K, vec v, vec tau, mat X, vec y, int kernal_type, mat x_test){
//   auto t1 = chrono::high_resolution_clock::now();
// 
//   int n = X.n_rows;
//   mat kx = kernal_compute_paralle(X, x_test, kernal_type);
// 
//   mat S_2 = diagmat(sqrt(tau)), I_n = eye(n, n);
//   mat B = I_n + S_2 * K * S_2;
//   mat L = chol(B, "lower");
//   mat L_1 = solve(L, I_n), Lt_1 = solve(L.t(), I_n);
// 
//   vec z = S_2 * Lt_1 * L_1 * S_2 * K * v;
// 
//   vec f_test = kx.t() * (v - z);
// 
//   mat v2 = L_1 * S_2 * kx;
//   vec Var = kernal_compute_diag(x_test, kernal_type) - sum(v2 % v2, 0).t();
// 
//   vec pai = normcdf(f_test / sqrt(1 + Var));
// 
//   auto t2 = chrono::high_resolution_clock::now();
//   chrono::duration<double, std::milli> fp_ms = t2 - t1;
//   cout << "(Paralle mat) The elapsed time to predict " << fp_ms.count()<< " milliseconds.\n";
// 
//   return pai;
// }

// vec kernal_compute_diag(const mat & X1, int kernal_type){
//   int n = X1.n_rows;
//   vec res(n);
// #pragma omp parallel for schedule(static)
//   for(int i = 0; i < n; i++){
//     res(i) = kernal_func_paralle(X1.row(i), X1.row(i), kernal_type);
//   }
//   
//   return res;
// }

#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]
using namespace Eigen;

// [[Rcpp::export]]
void modifyV1(MatrixXd x) {
  x(0,0) =  1;
}

// [[Rcpp::export]]
void modifyV2(Map<MatrixXd> x) {
  x(0,0) =  1;
}
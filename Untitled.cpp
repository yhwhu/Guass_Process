# include <RcppArmadillo.h>

using namespace arma;
using namespace std;


// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
int pp(mat A){
  cout << accu(A);
  return 0;
}

/*** R
pp(matrix(1:6, nrow = 2))
*/
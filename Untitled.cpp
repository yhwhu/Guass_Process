# include <RcppArmadillo.h>

using namespace arma;
using namespace std;


// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
int pp(mat A){
  vec c = {-1, -1, -1};
  A.diag() %= c;
  A.print();
  return 0;
}

/*** R
pp(matrix(c(2,3,6,7,22,6,7,10,2), nrow = 3))
*/
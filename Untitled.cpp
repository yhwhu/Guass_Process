# include <RcppArmadillo.h>

using namespace arma;
using namespace std;


// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
int pp(){
  int a = 2;
  switch (a){
  case 1:
    cout << "111" << endl;
    break;
  case 2:
    cout << "22" << endl;
    break;
  }
  return 0;
}

/*** R
pp()
*/
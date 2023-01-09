functions{
// ------------------------------------------------------
//     LINEAR PREDICTOR FOR THE LONGITUDINAL SUBMODEL                
// ------------------------------------------------------ 
    vector linear_predictor(vector x, vector times, int[] ID, vector beta, matrix bi){
         int N = num_elements(times);
         vector[N] out;

         out = beta[1] + bi[ID,1] + beta[2]*times + rows_dot_product(bi[ID,2],times) + beta[3]*x[ID];

         return out;
    } 
// ------------------------------------------------------ 
}


data{
  int N; 
  int n;
  int<lower=0> y[N];
  vector[N] times;
  int<lower=1,upper=n> ID[N];
  vector[n] x;
}


parameters{
  vector[3] beta;
  cov_matrix[2] Sigma;
  matrix[n,2] bi;
}


model{
// ------------------------------------------------------
//        LOG-LIKELIHOOD FOR LONGITUDINAL SUBMODEL                
// ------------------------------------------------------
{
   vector[N] logmu; 

   // Linear predictor
   logmu = linear_predictor(x, times, ID, beta, bi);

   // Longitudinal Poisson log-likelihood
   target += poisson_log_lpmf(y | logmu);
}  
// ------------------------------------------------------
//                       LOG-PRIORS                       
// ------------------------------------------------------
  // Longitudinal fixed effects
  target += normal_lpdf(beta | 0, 100);

  // Random-effects variance-covariance matrix
  Sigma ~ inv_wishart(2, diag_matrix(rep_vector(1.0,2)));

  // Random-effects
  for(i in 1:n){ target += multi_normal_lpdf(bi[i,1:2] | rep_vector(0.0,2), Sigma); }
 
}

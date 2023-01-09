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
  vector[n] Time;
  vector[n] status;
  vector[n] x;
  int K;
  vector[K] xk;
  vector[K] wk;
}


parameters{
  vector[3] beta;
  vector[2] gamma;
  real alpha;
  real<lower=0> phi;
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
//        LOG-LIKELIHOOD FOR SURVIVAL SUBMODEL                
// ------------------------------------------------------
{
  vector[n] haz;
  matrix[n,K] cumHazK;
  vector[n] cumHaz;

  for(i in 1:n){
       // Hazard function
       haz[i] = phi * pow(Time[i], phi-1) * exp( gamma[1] + gamma[2] * x[i] + alpha * exp(beta[1] + bi[i,1] + (beta[2] + bi[i,2])*Time[i] + beta[3] * x[i]) );

       // Hazard function evaluated at Gauss-Legendre quadrature integration points
       for(j in 1:K){
           cumHazK[i,j] = phi * pow(Time[i]/2*(xk[j]+1), phi-1) * exp( gamma[1] + gamma[2] * x[i] + alpha * exp(beta[1] + bi[i,1] + (beta[2] + bi[i,2])*Time[i]/2*(xk[j]+1) + beta[3] * x[i]) );
       }

       // Cumulative hazard function with Gauss-Legendre quadrature
       cumHaz[i] = Time[i] / 2 * dot_product(wk, cumHazK[i,]);
       
       target += status[i]*log(haz[i]) - cumHaz[i];
  }
}   
// ------------------------------------------------------
//                       LOG-PRIORS                       
// ------------------------------------------------------
  // Longitudinal fixed effects
  target += normal_lpdf(beta | 0, 100);

  // Survival fixed effects
  target += normal_lpdf(gamma | 0, 100);  
 
  // Association parameter
  target += normal_lpdf(alpha | 0, 100);

  // Shape parameter (Weibull hazard)
  target += inv_gamma_lpdf(phi | 0.1, 0.1);

  // Random-effects variance-covariance matrix
  Sigma ~ inv_wishart(2, diag_matrix(rep_vector(1.0,2)));

  // Random-effects
  for(i in 1:n){ target += multi_normal_lpdf(bi[i,1:2] | rep_vector(0.0,2), Sigma); }

}

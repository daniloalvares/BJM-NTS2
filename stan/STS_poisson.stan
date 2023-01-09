data{
  int n;
  vector[n] Time;
  vector[n] status;
  vector[n] x;
  vector[3] beta;
  matrix[n,2] bi;
  int K;
  vector[K] xk;
  vector[K] wk;
}


parameters{
  vector[2] gamma;
  real alpha;
  real<lower=0> phi;
}


model{
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
  // Survival fixed effects
  target += normal_lpdf(gamma | 0, 100);  
 
  // Association parameter
  target += normal_lpdf(alpha | 0, 100);

  // Shape parameter (Weibull hazard)
  target += inv_gamma_lpdf(phi | 0.1, 0.1);

}

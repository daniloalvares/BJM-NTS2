rm(list=ls(all=TRUE))

# Required sources
library(rstan)
library(rstanarm)
library(statmod)
library(JMbayes)

# Function for mode estimation
mode2 <- function(x){
  
  lim.inf <- min(x)-1
  lim.sup <- max(x)+1
  s <- density(x,from=lim.inf,to=lim.sup,bw=0.2)
  n <- length(s$y)
  v1 <- s$y[1:(n-2)]
  v2 <- s$y[2:(n-1)]
  v3 <- s$y[3:n]
  ix <- 1+which((v1<v2)&(v2>v3))
  md <- s$x[which(s$y==max(s$y))]
  
  return(md)
}

# Gauss-Legendre quadrature (15 points)
glq <- gauss.quad(15, kind = "legendre")
xk <- glq$nodes   # nodes
wk <- glq$weights # weights
K <- length(xk)   # K-points

# Number of patients
n <- nrow(prothros)
# Number of longitudinal observations per patient
M <- as.numeric(table(prothro$id))


###############################################
#      Joint specification (JS) approach      #
###############################################
# Setting initial values
init_fun1 <- function(...){ 
  list(beta=c(0,0,0), gamma=c(0,0), alpha=0, phi=1, Sigma=diag(c(1,1)), bi=matrix(0,nrow=n,ncol=2))
}

start_time1 <- Sys.time()
fitJS <- stan(file   = "stan/JS_poisson.stan", 
              data   = list(N=nrow(prothro), n=n, y=round(prothro$pro/10), times=prothro$time,
                            Time=prothros$Time, status=prothros$death, 
                            x=as.numeric(prothros$treat)-1, ID=rep(1:n,M),
                            K=K, xk=xk, wk=wk), 
              init   = init_fun1,
              warmup = 1000,                 
              iter   = 2000,                
              chains = 1,
              seed   = 1)
end_time1 <- Sys.time()
end_time1 - start_time1

print(fitJS)
###############################################

  
###############################################
#      Standard Two-Stage (STS) approach      #
###############################################
# Stage 1 - Longitudinal submodel
# Setting initial values
init_fun21 <- function(...){ 
  list(beta=c(0,0,0), Sigma=diag(c(1,1)), bi=matrix(0,nrow=n,ncol=2))
}

start_time21 <- Sys.time()
fitL <- stan(file   = "stan/LongTS_poisson.stan", 
             data   = list(N=nrow(prothro), n=n, y=round(prothro$pro/10), times=prothro$time,
                           x=as.numeric(prothros$treat)-1, ID=rep(1:n,M)),
             init   = init_fun21,
             warmup = 1000,                 
             iter   = 2000,                
             chains = 1,
             seed   = 1)
end_time21 <- Sys.time()
  
# Extracting random-effects
bi <- extract(fitL)$bi
bimode <- apply(bi,c(2,3),mode2)
  
# Extracting fixed-effects
beta1 <- unlist(extract(fitL, par="beta[1]"))
beta2 <- unlist(extract(fitL, par="beta[2]"))
beta3 <- unlist(extract(fitL, par="beta[3]"))
betamode <- c(mode2(beta1),mode2(beta2),mode2(beta3))
  
# Stage 2 - Survival submodel
# Setting initial values
init_fun22 <- function(...){ 
  list(gamma=c(0,0), alpha=0, phi=1)
}

start_time22 <- Sys.time()
fitSTS <- stan(file   = "stan/STS_poisson.stan", 
               data   = list(n=n, Time=prothros$Time, status=prothros$death,
                             x=as.numeric(prothros$treat)-1, 
                             beta=betamode, bi=bimode,
                             K=K, xk=xk, wk=wk),
               init   = init_fun22,
               warmup = 500,                 
               iter   = 1000,                
               chains = 1,
               seed   = 1)
end_time22 <- Sys.time()
end_time22 - start_time21

print(fitSTS)
###############################################
  

###############################################
#       Novel Two-Stage (NTS) approach        #
###############################################
# Extracting random-effects variance-covariance matrix
sigma2b1 <- mode2(unlist(extract(fitL, par="Sigma[1,1]")))
sigma2b2 <- mode2(unlist(extract(fitL, par="Sigma[2,2]")))
sigma2b12 <- mode2(unlist(extract(fitL, par="Sigma[1,2]")))
Sigmamode <- matrix(c(sigma2b1,sigma2b12,sigma2b12,sigma2b2),2,2)
  
# Stage 2 - Survival submodel
# Setting initial values
init_fun3 <- function(...){ 
  list(gamma=c(0,0), alpha=0, phi=1, bi=matrix(0,nrow=n,ncol=2))
}

start_time3 <- Sys.time()
fitNTS <- stan(file   = "stan/NTS_poisson.stan", 
               data   = list(N=nrow(prothro), n=n, y=round(prothro$pro/10), times=prothro$time,
                             Time=prothros$Time, status=prothros$death,
                             x=as.numeric(prothros$treat)-1, ID=rep(1:n,M),
                             beta=betamode, Sigma=Sigmamode,
                             K=K, xk=xk, wk=wk),
               init   = init_fun3,
               warmup = 500,                 
               iter   = 1000,                
               chains = 1,
               seed   = 1)
end_time3 <- Sys.time()
end_time3 - start_time3 + end_time21 - start_time21

print(fitNTS)
###############################################
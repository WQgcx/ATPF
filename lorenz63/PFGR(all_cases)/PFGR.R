library(dplyr) # A staple for modern data management in R
library(lubridate) # Useful functions for dealing with dates
library(ggplot2) # The preferred library for data visualisation
# library(tidync) # For easily dealing with NetCDF data
library(rerddap) # For easily downloading subsets of data
library(doParallel) # For parallel processing
library(magrittr)
library(ncdf4)
library(utils)
# library(tidync)
library(gganimate)
library(plot3D)
library(MASS)
library(Metrics) # for RMSE
library(RSpectra)
library(Matrix)
library(expm)
library(mvtnorm)
library(VGAM) # for laplace distribution
library(expm)


################################################################
############# new simulation for PFGR on L63 ###################
################################################################



Lorentz63 = function(ini_state,sigma,rho,beta,time,step){
  x = ini_state[1]
  y = ini_state[2]
  z = ini_state[3]
  
  
  all_state = c()
  for(t in c(1:time)){
    xt = x + step*sigma*(y-x)
    yt = y + step*(rho*x - y - x*z)
    zt = z + step*(x*y - beta*z)
    tmp_state = c(xt,yt,zt)
    all_state = rbind(all_state, tmp_state)
    x = xt
    y = yt
    z = zt
  }
  all_state
}



times = c(100,rep(2000,4))
obs_steps = c(1,1:4)

for(case_idx in c(1:5)){
  set.seed(case_idx)
  time1 = Sys.time()
  print(paste0('case: ',case_idx))
  filter_time = times[case_idx]
  obs_step = obs_steps[case_idx]
  
  step = 0.02 # 0.02 0.05
  theta = 1
  t0 = 0
  N = 1000
  
  # ini_state = c(9,16,6)
  sigma = 10
  rho = 28
  beta = 8/3
  
  # result = rk4(L63,t0,time,ini_state,step)
  true_state = read.csv(paste0("x_T=",filter_time,'_step=',obs_step,'.csv'),header = F)
  # true_state = result$target
  
  colnames(true_state) = c('X','Y','Z')
  # scatter3D(x = true_state[,1],y = true_state[,2],z = true_state[,3],theta = 60, phi = 20,
  #           type = 'p',colvar = c(1:nrow(true_state)))
  
  
  # create observation
  obs_time = seq(1,nrow(true_state)-1,by = 1)
  H = diag(x = c(1,1,1)) # observation matrix
  R = diag(theta^2,nrow = 3) # observation error
  obs_state = read.csv(paste0("y_T=",filter_time,'_step=',obs_step,'.csv'),header = F)
  ini_obs = obs_state[1,]
  obs_state = obs_state[-1,]
  
  # obs_state = c()
  # for( i in c(1:length(obs_time))){
  #   tmp_true = true_state[obs_time[i],]
  #   
  #   tmp_obs = as.vector(H%*%tmp_true + mvrnorm(1,rep(0,3),R))
  #   # tmp_obs = as.vector(H%*%tmp_true +  rlaplace(3,0,theta))
  #   obs_state = rbind(obs_state,tmp_obs)
  # }
  # 
  
  
  # create ensemble initial state and ensemble observation
  # N = 100 # ensemble size
  ens_ini_state = list()
  current_state = c()
  
  for( i in c(1:N)) {
    tmp =  mvrnorm(1,rep(0,3),R)
    current_state = rbind(current_state,tmp)
    ens_ini_state[[i]] = tmp
  }
  
  
  # assimilate the initial state
  
  obs_y = as.numeric(ini_obs)
  f_vec = apply(current_state,1,function(x){dmvnorm(obs_y  - x,mean = c(0,0,0),sigma = R)  })
  # sum_prob = c(sum_prob,sum(f_vec))
  f_vec = f_vec/sum(f_vec)
  
  M = diag(f_vec) - f_vec %*% t(f_vec)
  svd_M = svd(M)
  V = svd_M$u
  L = diag(sqrt(svd_M$d))
  m = nrow(L)
  
  mean_state = apply(current_state,2,weighted.mean,w = f_vec)
  perturb_state = t(current_state) %*% V %*% L
  post_state = perturb_state %*% matrix(rnorm(m*N,mean = 0, sd = 1),nrow = m,ncol = N) + matrix(rep(mean_state,N),ncol = N)
  post_mean = apply(post_state,1,mean)
  #inflation
  post_state = matrix(rep(post_mean,N),ncol = N) + (1+0.02)*(post_state - matrix(rep(post_mean,N),ncol = N))
  
  for(ens_idx in c(1:N)){
    curr_state = post_state[,ens_idx]
    ens_ini_state[[ens_idx]] = curr_state
  }
  
  
  ##########################################################
  # forward process: particle filter with posterior Gaussian
  ##########################################################
  
  ens_state_a_PFGR= list()
  ens_state_f_PFGR = list()
  curr_time = 0
  sum_prob = c()
  for( obs_idx in c(1:filter_time) ){
    # obs_idx = 1
    if(obs_idx %% 100 == 0) {print(obs_idx)}
    target_time = obs_time[obs_idx]
    obs_y = obs_state[obs_idx,]
    current_state = c()
    # target_obs = obs_state[i,]
    
    for( ens_idx in c(1:N)){ #update each ensemble run to the current obs time
      # ens_idx = 1
      if(curr_time ==  0){
        curr_state = ens_ini_state[[ens_idx]]
      }else{
        curr_state = ens_state_a_PFGR[[ens_idx]][curr_time,]
      }
      
      
      #only need the exact current state 
      model_state = Lorentz63(curr_state,sigma,rho,beta,(target_time - curr_time)*obs_step,step)
      current_state = rbind(current_state,model_state[nrow(model_state),])
      
      if(curr_time == 0){
        ens_state_f_PFGR[[ens_idx]] = matrix(model_state[nrow(model_state),],nrow = 1)
        ens_state_a_PFGR[[ens_idx]] = matrix(model_state[nrow(model_state),],nrow = 1)
      }else{
        ens_state_f_PFGR[[ens_idx]] = rbind(ens_state_f_PFGR[[ens_idx]],model_state[nrow(model_state),])
        ens_state_a_PFGR[[ens_idx]] = rbind(ens_state_a_PFGR[[ens_idx]],model_state[nrow(model_state),])
      }
    }
    
    # update particle by PG    
    
    f_vec = apply(current_state,1,function(x){dmvnorm(obs_y  - x,mean = c(0,0,0),sigma = R)  })
    # f_vec = apply(current_state,1,function(x){prod(dlaplace(obs_y  - x,0,theta))  })
    
    # sum_prob = c(sum_prob,sum(f_vec))
    f_vec = f_vec/sum(f_vec)
    M = diag(f_vec) - f_vec %*% t(f_vec)
    svd_M = svd(M)
    
    # d = svd_M$d
    # m = which(cumsum(d) >= 0.9*sum(d))[1]
    # if(m == 1){m = 2}
    # V = svd_M$u[,1:m]
    # L = diag(sqrt(svd_M$d[1:m]))
    
    V = svd_M$u
    L = diag(sqrt(svd_M$d))
    m = nrow(L)
    
    mean_state = apply(current_state,2,weighted.mean,w = f_vec)
    perturb_state = t(current_state) %*% V %*% L
    # mean_state = apply(perturb_state,1,mean)
    post_state = perturb_state %*% matrix(rnorm(m*N,mean = 0, sd = 1),nrow = m,ncol = N) + matrix(rep(mean_state,N),ncol = N)
    post_mean = apply(post_state,1,mean)
    # var(t(post_state))
    # t(current_state) %*% M %*% current_state
    
    #inflation
    post_state = matrix(rep(post_mean,N),ncol = N) + (1+0.02)*(post_state - matrix(rep(post_mean,N),ncol = N))
    # print(sum(diag(var(t(post_state)))))
    
    for(ens_idx in c(1:N)){
      curr_state = post_state[,ens_idx]
      ens_state_a_PFGR[[ens_idx]][nrow(ens_state_a_PFGR[[ens_idx]]),] = curr_state
    }
    
    curr_time = target_time
    
    
  }
  
  
  esti_state_PFGR = matrix(0,nrow = filter_time + 1,ncol = 3)
  for(ens_idx in c(1:N)){esti_state_PFGR = esti_state_PFGR + rbind(ens_ini_state[[ens_idx]],ens_state_a_PFGR[[ens_idx]])}
  esti_state_PFGR = esti_state_PFGR/N
  
  
  png(filename = paste0("PFGR T = ",filter_time," step = ",obs_step,".png"), 
      width = 600, height = 600)
  scatter3D(x = esti_state_PFGR[,1],y = esti_state_PFGR[,2],z = esti_state_PFGR[,3],theta = 60, phi = 20,
            type = 'l',col = 'red',ticktype = "detailed" )
  lines3D(x = true_state[,1],y = true_state[,2],z = true_state[,3]
          , col = "black", add = TRUE)
  legend("topright", legend = c("Filtered Data", "Real Data"), col = c("red", "black"), lty = 1, cex = 0.8)
  title(paste0("PFGR T = ",filter_time," step = ",obs_step))
  dev.off()
  
  # calculate difference
  diff_state_PFGR = esti_state_PFGR - true_state
  total_err_PFGR = apply(diff_state_PFGR,1,function(x) mean(abs(x)))
  rmse_PFGR = apply(diff_state_PFGR,1,function(x) sqrt(sum(x^2)))
  plot(rmse_PFGR,type = 'l')
  # sum(rmse_PFGR)/sqrt(3)/filter_time
  
  
  
  # plot(diff_state_PFGR[,1],col = 'red',ylim = c(-10,10))
  # lines(diff_state_PFGR[,2],col = 'blue')
  # lines(diff_state_PFGR[,3],col = 'black')
  
  # calculate variance of three position
  # ens_spread = c()
  
  # for( t in c(1:nrow(ens_state_a_PFGR[[1]]))){
  #   # t = 1
  #   state = c()
  #   for( ens_idx in c(1:N)){
  #     state = rbind(state, ens_state_a_PFGR[[ens_idx]][t,])
  #   }
  #   var = sum(sqrt(diag(var(state))))/sqrt(3)
  #   # all_var = rbind(all_var,var)
  #   ens_spread = c(ens_spread,var)
  # }
  # mean(ens_spread)
  
  # plot(all_var[,1],col = 'red')
  # lines(all_var[,2],col = 'blue')
  # lines(all_var[,3],col = 'black')
  
  # cover_id = c()
  # for( t in c(1:nrow(ens_state_a_PFGR[[1]]))){
  #   # t = 1
  #   state = c()
  #   truth = true_state[t+1,3]
  #   for( ens_idx in c(1:N)){
  #     state = c(state, ens_state_a_PFGR[[ens_idx]][t,3])
  #   }
  #   qq = quantile(state,probs = c(0.025,0.975))
  #   id = as.numeric(truth <= qq[2] & truth >= qq[1]) 
  #   cover_id = c(cover_id,id)
  # }
  # tmp_result = c(filter_time,obs_step,sum(rmse_PFGR)/sqrt(3)/filter_time,  mean(ens_spread), mean(cover_id))
  tmp_result = c(filter_time,obs_step,sum(rmse_PFGR)/sqrt(3)/filter_time)
  print(tmp_result)
  write.csv(esti_state_PFGR,paste0("T=",filter_time,"_step=",obs_step,'.csv')) 
}
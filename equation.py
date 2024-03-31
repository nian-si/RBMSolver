import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as normal
from tqdm import tqdm
import sys
import time


class Equation(object):
    """Base class for defining PDE related function."""
    def __init__(self, eqn_config):
        
        
        self.dim = eqn_config.dim
        self.gamma = eqn_config.discount
        self.x0 = eqn_config.x0
        self.var = 1
        if "var" in eqn_config:
            self.var = eqn_config.var
        self.rho = eqn_config.rho
        self.a = eqn_config.a
        if isinstance(eqn_config.R, list) is False and np.abs(eqn_config.R)  < 1e-9 and np.abs(self.rho) < 1e-9:
            self.rep_flag = True  
        else:
            self.rep_flag = False

    
   


    def gen_samples(self, num_sample,T,N, Total_iterations,  simulation_method):
        delta_t = T / N 
        R = self.R
        sqrt_delta_t = np.sqrt(delta_t)
        dw_sample = np.random.normal(size=[num_sample, self.dim, N * Total_iterations])# * sqrt_delta_t
        dw_sigma_t = np.zeros(dw_sample.shape)
        x0 = np.ones(self.dim) * self.x0;
        
        y_i = np.tile(x0, num_sample).reshape([num_sample, len(x0)])

        y_sample_array = []
        

        for i in range(N * Total_iterations):
            if i % 1000 == 0:
                print("Generate Samples:",i)

            
            if i % N == 0:

                if simulation_method == "fixed":
                    y_i = np.tile(x0, num_sample).reshape([num_sample, len(x0)])
                elif simulation_method == "uniform":
                    y_i = np.random.uniform (size = (num_sample, len(x0))) * 2
                y_sample_array.append(y_i)

            dw_sigma_t[:,:,i] = self.diffusion(dw_sample[:, :, i]) * sqrt_delta_t
           

            delta_x = self.drift() * delta_t + dw_sigma_t[:,:,i]

            x_i = y_i + delta_x
            y_i = self.Skorokod(x_i, R)
            y_sample_array.append(y_i)
            


        return np.stack(y_sample_array, axis = 2), dw_sigma_t
        
    def Skorokod(self,X_full, R):
        
        eps = 1e-8
        num_samples = X_full.shape[0]
        Y_ret = np.zeros(X_full.shape)
        for i in range(num_samples):
            X = X_full[i,:]
            Y = X
            while sum (Y < -eps) >0:
                base = (Y < eps)
                R_b = R[base][:,base]

                if len(R_b) > 0:
                    L_b = - np.linalg.solve(R_b, X[base])
                    Y = X + R[:, base] @ L_b
            Y_ret[i,:] = Y
        return Y_ret
 

    def w_tf(self, x):
        """Running cost in control problems."""
        raise NotImplementedError

        
    def V_true(self, x):
        """True value function"""
        raise NotImplementedError

    def V_grad_true(self, x):
        raise NotImplementedError

    def c_true(self):
        """True cost"""
        raise NotImplementedError
    
        
    def sigma(self): #num_sample x dim x dim_w
        """diffusion coefficient"""
        raise NotImplementedError
        
    def drift(self):
        """drift in the SDE"""
        raise NotImplementedError
    
    def diffusion(self,  dw):
        """diffusion in the SDE"""
        raise NotImplementedError





class dynamicPricing(Equation): 
    def __init__(self, eqn_config):
        super(dynamicPricing, self).__init__(eqn_config)
        self.name = 'dynamicPricing'
        R = np.identity(self.dim)
        if "queue_example" not in eqn_config: 
            R[1:,0] = eqn_config.R
        elif eqn_config.queue_example == "1":
            R[1:,0] = eqn_config.R
            
        elif eqn_config.queue_example == "2":
            R[0,1:] = eqn_config.R
        self.R = R

        self.mu = np.repeat(eqn_config.mu,self.dim)
      

        
        if "cTrue" in eqn_config:
            self.cTrue = eqn_config.cTrue
        if "constOptimal" in eqn_config:
            self.constOpt = eqn_config.constOptimal
        if "linearOptimal" in eqn_config:
            self.linearOpt = eqn_config.linearOptimal
        
        if self.rep_flag:
            self.h = np.repeat(eqn_config.h,self.dim)
        else:
            if "queue_example" not in eqn_config: 
                self.h = np.repeat(eqn_config.h * 0.95,self.dim) 
                self.h[0] = eqn_config.h
            elif eqn_config.queue_example == "1":
                self.h = np.repeat(eqn_config.h * 0.95,self.dim) 
                self.h[0] = eqn_config.h
                
            elif eqn_config.queue_example == "2":
                self.h = np.repeat(eqn_config.h,self.dim) 
                self.h[0] = eqn_config.h * 0.95
        
    def w_tf(self, x, grad_t, a_lowbound  = 0): #num_sample * 1
        w = tf.linalg.matvec(x,self.h)
        w = tf.reshape(w, [-1,1])
        w = w - tf.reduce_sum(tf.math.square(grad_t), 1, keepdims=True) / 4
        return w,0



    def sigma(self): # x is num_sample x dim, u is num_sample x dim_u, sigma is num_sample x dim x dim_w
       
        mat1 = np.identity(self.dim)
        for i in np.arange(1,self.dim):
            for j in np.arange(1,self.dim):
                if i!=j:
                    mat1[i,j] = self.rho
        return np.linalg.cholesky(mat1)


    
    def drift(self):
        return self.mu
    
    def diffusion(self,  dw): #sigma num_sample x dim x dim_w, dw is num_sample x dim_w
        return np.dot(self.sigma(), dw.transpose()).transpose() # num_sample x dim
    
    def const_control_optimal(self):  #1d case
        try:
            return self.constOpt * self.dim; 
        except:
            print("NO CONST CONTROL PROVIDED!!")
            return 0
    
    def linear_control_optimal(self): #1d case
        try:
            return self.linearOpt  * self.dim
        except:
            print("NO LINEAR CONTROL PROVIDED!!")
            return 0
    def V_true(self, z):
        print("NO KNOWN V TRUE!!")
        return -1
    def c_true(self):  #1d case
        try:
            return self.cTrue  * self.dim;   
        except:
            print("NO OPTIMAL VALUE PROVIDED!!")
            return 0

class thinStream(Equation): 
    def __init__(self, eqn_config):
        super(thinStream, self).__init__(eqn_config)   
        self.name = 'thinStream'
        R = np.identity(self.dim)
        if "queue_example" not in eqn_config: 
            R[1:,0] = eqn_config.R
        elif eqn_config.queue_example == "1":
            R[1:,0] = eqn_config.R
            
        elif eqn_config.queue_example == "2":
            R[0,1:] = eqn_config.R
        self.R = R  

 
        
        self.mu = np.repeat(eqn_config.mu,self.dim)
        

        self.v = np.repeat(eqn_config.v,self.dim)


        
        if "cTrue" in eqn_config:
            self.cTrue = eqn_config.cTrue
        
        
        if self.rep_flag:
            
            self.h = np.repeat(eqn_config.h,self.dim)
            
        else:
            if "queue_example" not in eqn_config: 
                self.h = np.repeat(eqn_config.h * 0.95,self.dim) 
                self.h[0] = eqn_config.h
            elif eqn_config.queue_example == "1":
                self.h = np.repeat(eqn_config.h * 0.95,self.dim) 

                self.h[0] = eqn_config.h
                
            elif eqn_config.queue_example == "2":
                self.h = np.repeat(eqn_config.h,self.dim) 
                self.h[0] = eqn_config.h * 0.95
        print(self.h)        

            
        
    def w_tf(self, x, grad_t, a_lowbound  = 0): #num_sample * 1
        w = tf.linalg.matvec(x,self.h)
        w = w - tf.linalg.matvec(grad_t,self.mu)
        w = tf.reshape(w, [-1,1])
        max_zero_grad = tf.math.maximum(tf.cast(0., tf.float64) , grad_t - self.v )
        min_zero_grad = tf.math.minimum(tf.cast(0., tf.float64) , grad_t - self.v )

        w = w - tf.reduce_sum(max_zero_grad, 1, keepdims=True) * (self.a)
        w = w - tf.reduce_sum((min_zero_grad), 1, keepdims=True) * a_lowbound

        


        
        #zero_grad = tf.math.minimum(tf.cast(0., tf.float64),grad_t ) 
        
        
        return w,  0



    def sigma(self): # x is num_sample x dim, u is num_sample x dim_u, sigma is num_sample x dim x dim_w
        mat1 = np.identity(self.dim)
        if self.rho > 1.0:
            for i in np.arange(1,self.dim):
                for j in np.arange(1,self.dim):
                    if i!=j:
                        mat1[i,j] = -self.R[i,0] * self.R[j,0]
            return np.linalg.cholesky(mat1)

        for i in np.arange(1,self.dim):
            for j in np.arange(1,self.dim):
                if i!=j:
                    mat1[i,j] = self.rho
        return np.linalg.cholesky(mat1)

        
    
    def drift(self):
        return self.mu
    
    def diffusion(self,  dw): #sigma num_sample x dim x dim_w, dw is num_sample x dim_w
        return np.dot(self.sigma(), dw.transpose()).transpose() # num_sample x dim
    
    def c_true(self):
        if self.rep_flag:
            sigma =  self.sigma()[0,0];
            return np.sqrt(self.h[0]) * sigma * np.sqrt(4 * self.a ** 2 * self.v[0] + self.h[0] * sigma ** 2) / 2 / self.a * self.dim
        else:
            return 0;
    def V_true(self, z):
        h = self.h[0]
        v = self.v[0]
        a = self.a
        sigma =  self.sigma()[0,0];
        gamma = (sigma * np.sqrt(h) * np.sqrt(4 * a ** 2  * v + h * sigma ** 2)) / (2 * a)
        zstar = (2 * a * gamma - h * sigma ** 2) / (2 * a * h)
        ans = np.zeros(z.shape)
       
        ans[z < zstar] = (-h * z[z < zstar] **3/3  +  z[z < zstar] **2 * gamma) / (sigma ** 2)
        ans[z >= zstar] = v * z[z >= zstar] + h * z[z >= zstar] ** 2 / 2 / a + z [z >= zstar] * (-2 *a *gamma + h *sigma  ** 2) / (2 * a** 2)

        offset_constant = 1/6 * (-6 * v * gamma /h + 4 * gamma ** 3 / (h **2 * sigma ** 2) + 3 * (a * v - gamma) * sigma ** 2 / a ** 2 + h * sigma **4 /a **3)
        ans[z >= zstar] = offset_constant + ans[z >= zstar]

        return tf.math.reduce_sum(ans, axis = 1, keepdims = 1)
        
    def V_grad_true(self, z): 
        h = self.h[0]
        v = self.v[0]
        a = self.a
        sigma =  self.sigma()[0,0];
        gamma = (sigma * np.sqrt(h) * np.sqrt(4 * a ** 2  * v + h * sigma ** 2)) / (2 * a)
        
        zstar = (2 * a * gamma - h * sigma ** 2) / (2 * a * h)
        ans = np.zeros(z.shape)
        ans[z < zstar] = (-h * z[z < zstar]  ** 2 + 2 * z[z < zstar]  * gamma)  / (sigma ** 2)
        ans[z >= zstar] = (2 * a ** 2 * v + 2 * a * h * z[z >= zstar] - 2 * a * gamma + h * sigma ** 2) / (2 * a ** 2)
        
        return ans

    def const_control_optimal(self):   #1d case
        if self.rep_flag:
            sigma =  self.sigma()[0,0];
            return sigma * np.sqrt(2 * self.h[0] * self.v[0])  * self.dim
        else:
            return 0
    
    def linear_control_optimal(self):  #1d case
        if self.rep_flag:
            sigma =  self.sigma()[0,0];
            return 2 *  sigma * np.sqrt(self.h[0] * self.v[0] / np.pi) * self.dim
        else:
            return 0
           

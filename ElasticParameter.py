import numpy as np
import matplotlib.pyplot as plt

def velocities(mu,lamda,rho): 
    vs = np.sqrt(mu/rho)
    vp = np.sqrt((lamda+2*mu)/rho)
    return [vp,vs]
def mu_lamda(E,K,sigma): 
    mu = 3*K*E/(9*K-E)
    lamda = 3*K*(3*K-E)/(9*K-E)
    return [mu,lamda]
def K_sig_E(lamda,mu):
    E = mu*((3*lamda+2*mu)/(lamda+mu))
    sigma = lamda/(2*(lamda+mu))
    K = lamda+2/3*mu
    return [E,sigma,K]
    
    
def elastic_moduli(vp=np.nan,vs=np.nan,rho=np.nan,E=np.nan,sigma=np.nan,K=np.nan,mu=np.nan,lamda=np.nan):
    if (~np.isnan(E) & ~np.isnan(sigma)):#1
        K = E/(3*(1-2*sigma))
        [mu,lamda] = mu_lamda(E,K,sigma)
        return np.array([E,sigma,K,mu,lamda,*velocities(mu,lamda,rho),rho])
    
    if (~np.isnan(E) & ~np.isnan(K)): #2
        sigma = (3*K-E)/(6*k)
        [mu,lamda] = mu_lamda(E,K,sigma)
        return np.array([E,sigma,K,mu,lamda,*velocities(mu,lamda,rho),rho])
    
    if (~np.isnan(E) & ~np.isnan(mu)): #3
        sigma = (E-2*mu)/(2*mu)
        K = (mu*E)/(3*(3*mu-E))
        [mu,lamda] = mu_lamda(E,K,sigma)
        return np.array([E,sigma,K,mu,lamda,*velocities(mu,lamda,rho),rho])
    
    if (~np.isnan(sigma) & ~np.isnan(K)): #4
        E = 3*K*(1-2*sigma)
        [mu,lamda] = mu_lamda(E,K,sigma)
        return np.array([E,sigma,K,mu,lamda,*velocities(mu,lamda,rho),rho]) 
    
    if (~np.isnan(sigma) & ~np.isnan(mu)): #5
        lamda = mu*(2*sigma/(1-2*sigma))
        [E,sigma,K] = K_sig_E(lamda,mu)
        return np.array([E,sigma,K,mu,lamda,*velocities(mu,lamda,rho),rho])
    
    if (~np.isnan(sigma) & ~np.isnan(lamda)): #6 
        mu = lamda*((1-2*sigma)/(2*sigma))
        [E,sigma,K] = K_sig_E(lamda,mu)
        return np.array([E,sigma,K,mu,lamda,*velocities(mu,lamda,rho),rho])
    
    if (~np.isnan(K) & ~np.isnan(mu)):#7
        lamda = K-2*mu/3
        [E,sigma,K] = K_sig_E(lamda,mu)
        return np.array([E,sigma,K,mu,lamda,*velocities(mu,lamda,rho),rho])
    
    if (~np.isnan(K) & ~np.isnan(lamda)):#8
        mu = 1.5*(K-lamda)
        [E,sigma,K] = K_sig_E(lamda,mu)
        return np.array([E,sigma,K,mu,lamda,*velocities(mu,lamda,rho),rho])
    
    if (~np.isnan(mu) & ~np.isnan(lamda)):#9
        [E,sigma,K] = K_sig_E(lamda,mu)
        return np.array([E,sigma,K,mu,lamda,*velocities(mu,lamda,rho),rho])
    
    if (~np.isnan(vp) & ~np.isnan(vs) & ~np.isnan(rho)):#10
        mu = rho*vs**2
        lamda = rho*vp**2 - 2*mu
        [E,sigma,K] = K_sig_E(lamda,mu)
        return np.array([E,sigma,K,mu,lamda,*velocities(mu,lamda,rho),rho])
    
    print('done ...')
    print('go to next cell')
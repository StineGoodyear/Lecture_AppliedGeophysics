import numpy as np
import scipy.special as scp
import matplotlib.pyplot as plt
import time
import math

def precompute_bessel_function(L_model,dx,n_max):
    vec = np.arange(0,n_max+dx,dx); 
    bessel = np.zeros([len(L_model),len(vec)])
    for jdx,L in enumerate(L_model):
        for idx,lamb in enumerate(vec):
            bessel[jdx,idx] = scp.jv(1,lamb*L)*lamb
    return bessel

def forward_model_1(rho,d,L_model,bessel,dx,n_max): 
    ''' Berechnet die scheinbaren spezifischen Widerstände für Schichten mit der
        Dicke d und spezifischen Widerständen rho für bestimmte Auslagenlängen L/2 

    © Nepomuk Boitz, Mai 2020, boitz@geophysik.fu-berlin.de
'''
    ## Initialize parameters
    n = len(rho); 
    K = np.zeros([n]); 
    T = np.zeros([n])
    vec1 = np.arange(0,n_max+dx,dx); 
    T_star = np.zeros(len(vec1))
    P = np.zeros([n-1])
    rho_a = np.zeros([len(L_model)])
    temp = np.zeros([n-1,len(vec1)])
    check = np.zeros([n-1,len(vec1)])
    K[n-1] = 1; 
    # Precomputations
    for i in range(0,n-1):
        P[i] = rho[i]/rho[i+1]; 
        for idx,lamb in enumerate(vec1):
            temp[i,idx] = np.tanh(d[i]*lamb)

    vec = np.arange(n-2,-1,-1)
    count = 0; 
    for idx,lamb in enumerate(vec1):  
        for i in vec:
            K[i] = (K[i+1]+P[i]*temp[i,idx])/(P[i] + K[i+1]*temp[i,idx]);
            check[i,idx] = K[i]
        T_star[idx] = rho[0]*check[0,idx];
    
    
    # Forward Modeling     
    for jdx,L in enumerate(L_model):
        fac = 0;
        for idx,lamb in enumerate(vec1):
            fac = fac + (((T_star[idx]-rho[0])*bessel[jdx,idx])*dx);
        fac = rho[0]+L**2*fac; 
        rho_a[count] = fac;   
        count += 1; 

    return rho_a

def invert_model(n_realisation,it_max,start_model,data,dx,n_max,fac):
    history = np.zeros([n_realisation,len(start_model)+1])
    bessel = precompute_bessel_function(data[:,0],dx,n_max)
    n_layer = len(start_model)//2+1
    t_start = time.time()
    for j in range(n_realisation):
        rho = start_model[range(0,len(start_model)//2+1)]; 
        d = start_model[range(len(start_model)//2+1,len(start_model))]; 
        
        rho_a = forward_model_1(rho,d,data[:,0],bessel,dx,n_max);
        if j == 0:
            plot_single_model(data,rho_a)
            plt.title('Datenanpassung des Startmodells')
            
        misfit = np.linalg.norm(np.log10(rho_a)  - np.log10(data[:,2]),2); 
        mis = np.zeros(it_max)
        a = time.time()
        for i in range(it_max):
            rho_update = np.random.randn(n_layer)*fac*rho;
            d_update =np.random.randn(n_layer-1)*fac*d;
            rho_a = forward_model_1(rho+rho_update,d+d_update,data[:,0],bessel,dx,n_max);
            mis[i] = np.linalg.norm(np.log10(rho_a)  - np.log10(data[:,2]),2);
            if (mis[i] < misfit):
                misfit = mis[i];
                rho  = rho+rho_update;
                d = d+d_update;
        history[j,:] = np.concatenate([rho, d,np.array( [mis[i]])]);
        t_end = time.time()
        print('Berechnet Modell %d von insgesamt %d Modellen, durchschnittliche Berechnungsgzeit pro Modell %.2f s' % (j+1,n_realisation,(t_end-t_start)/(j+1)))
    return history

def plot_single_model(data,rho_a): 
    
    plt.loglog(data[:,0],data[:,2],'b.-')
    plt.loglog(data[:,0],rho_a,'r.-')
    plt.xlabel('L/2 [m]')
    plt.ylabel('scheinbarer spezifischer Widerstand [Ohm m]')
    plt.legend(['Data','Modell-Fit'])
    plt.title('Datenanpassung durch das beste invertierte Modell')
    
def plot_equivalent_models(history):
    #fig, ax = plt.subplots(figsize=(8, 8))

    n_layer = np.size(history,1)//2
    for i in range(np.size(history,0)):
        z = np.zeros(2*n_layer)
        rho = np.zeros(2*n_layer)
        for j in range(n_layer):
            rho[range(j*2,j*2+2)] = history[i,j]
        for j in range(n_layer-1):
            z[range(j*2+1,j*2+3)] = history[i,n_layer+j] + z[j]
        z[n_layer*2-1] = 20
        plt.loglog(rho,z,'k--')
        if np.argmin(history[:,np.size(history,1)-1])==i:
            plt.loglog(rho,z,'r-')
    plt.ylim((0.1,50))  
    plt.xlim((0.1,500))   
    plt.gca().invert_yaxis()
    plt.ylabel('Tiefe [m]')
    plt.xlabel('Spezifischer Widerstand [Ohm m]')
    plt.title('Äquivalenzmodelle')
    
    
def plot_final_model(history):

    n_layer = np.size(history,1)//2
    for i in range(np.size(history,0)):
        z = np.zeros(2*n_layer)
        rho = np.zeros(2*n_layer)
        for j in range(n_layer):
            rho[range(j*2,j*2+2)] = history[i,j]
        for j in range(n_layer-1):
            z[range(j*2+1,j*2+3)] = history[i,n_layer+j] + z[j]
        z[n_layer*2-1] = 20
        #plt.loglog(rho,z,'k--')
        if np.argmin(history[:,np.size(history,1)-1])==i:
            plt.loglog(rho,z,'r-')
    plt.ylim((0.1,50))    
    plt.xlim((0.1,500))
    plt.gca().invert_yaxis()
    plt.ylabel('Tiefe [m]')
    plt.xlabel('Spezifischer Widerstand [Ohm m]')
    plt.title('Das beste invertierte Modell')
    
def plot_start_model(rho_start,d_start,d_max):
    
    model = np.zeros([len(rho_start)*2,2])
    ds = []
    for i in range(len(d_start)):
        ds.append(d_start[i])
    ds.append(d_max)

    numi = 0
    for i in range(0,np.size(model,0),2):
        model[i,1] = rho_start[numi]
        model[i+1,1] = rho_start[numi]
        model[i+1:i+3,0] = ds[numi]
        numi += 1
    plt.ylim((0.1,50))   
    plt.xlim((0.1,500))
    plt.loglog(model[:,1],model[:,0],'-r')
    plt.gca().invert_yaxis()
    plt.ylabel('Tiefe [m]')
    plt.xlabel('spezifischer Widerstand [Ohm m]')
    plt.title('Startmodell')   
    
def plot_sondierungskurve(L_halbe,rho_a):

    plt.loglog(L_halbe,rho_a,'-b')
    plt.xlabel('L/2 [m]')
    plt.ylabel('scheinbarer spezifischer Widerstand [Ohm m]')
    plt.title('Sondierungskurve')
    
        

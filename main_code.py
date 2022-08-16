import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
import scipy as scipy

def sample_power_law(gamma = 2, kappa_0 = 1, sample_size = 10):
    
    #sampling independent uniform random variables
    uniform_samples = np.random.uniform(size = sample_size)
    
    #applying inversion to obtain i.i.d.samples from a power-law distribution
    samples = kappa_0 * ( uniform_samples ** (1 / (1 - gamma)) )
    return samples
        
def distance(theta, phi, R):
    return R*(np.pi - abs(np.pi - abs(theta - phi)))

def distance_scale(kappa, Lambda, mu):
    return mu*kappa*Lambda

def connection_power_law(x, beta_parameter):
    return 1/(1+x**beta_parameter)

def latent_bipartite_graph( N, M, connection_parameter, kappa_distr = ('power_law', 2.5, 1), \
                 lambda_distr = ('power_law', 2.5, 1), kappa = None, Lambda = None, R = 1, mu = 1):
        
    thetas = 2*np.pi * np.random.uniform(size = N)
    
    if kappa != None:
        kappas = np.repeat(kappa, N)
        
    else:
        if kappa_distr[0] == 'power_law':
            kappas = sample_power_law(gamma = kappa_distr[1], kappa_0 = kappa_distr[2], \
                                       sample_size = N)       
        elif kappa_distr[0] == 'poisson':
            kappas = np.repeat(kappa_distr[1], N)
        else:
            raise ValueError('Kappa distribution not supported')
        
        
    phis = 2*np.pi * np.random.uniform(size = M)
    
    if Lambda != None:
        lambdas = np.repeat(Lambda, M)
    
    else:
        if lambda_distr[0] == 'power_law':
            lambdas = sample_power_law(gamma = lambda_distr[1], kappa_0 = lambda_distr[2], \
                                       sample_size = M)    
        elif lambda_distr[0] == 'poisson':
            lambdas = np.repeat(lambda_distr[1], M)        
        else:
            raise ValueError('Lambda distribution not supported')
            
    #creating the graph and the nodes
    G = nx.Graph()
    # Add nodes with the node attribute "bipartite"
    G.add_nodes_from(range(N), bipartite=0)
    G.add_nodes_from(range(N, N+M), bipartite=1)
    
    for i in range(N):
        if N/(i+1) in range(11):
            print(i)
        for j in range(M):
            if np.random.uniform() <= \
                connection_power_law(distance(thetas[i], phis[j], R) / \
                                     distance_scale(kappas[i], lambdas[j], mu),\
                                         connection_parameter):
                G.add_edge(i, N+j)
    return G
                
class LatentBipartiteGraph():
    def __init__(self, N, M, connection_parameter, kappa_distr = ('power_law', 2.5, 1), \
                 lambda_distr = ('power_law', 2.5, 1), kappa = None, Lambda = None, R = 1, mu = 1):
        
        thetas = 2*np.pi * np.random.uniform(size = N)
        
        if kappa != None:
            kappas = np.repeat(kappa, N)
            
        else:
            if kappa_distr[0] == 'power_law':
                kappas = sample_power_law(gamma = kappa_distr[1], kappa_0 = kappa_distr[2], \
                                           sample_size = N)       
            elif kappa_distr[0] == 'poisson':
                kappas = np.repeat(kappa_distr[1], N)
            else:
                raise ValueError('Kappa distribution not supported')
            
            
        phis = 2*np.pi * np.random.uniform(size = M)
        
        if Lambda != None:
            lambdas = np.repeat(Lambda, M)
        
        else:
            if lambda_distr[0] == 'power_law':
                lambdas = sample_power_law(gamma = lambda_distr[1], kappa_0 = lambda_distr[2], \
                                           sample_size = M)    
            elif lambda_distr[0] == 'poisson':
                lambdas = np.repeat(lambda_distr[1], M)        
            else:
                raise ValueError('Lambda distribution not supported')
                
        #creating the graph and the nodes
        G = nx.Graph()
        # Add nodes with the node attribute "bipartite"
        G.add_nodes_from(range(N), bipartite=0)
        G.add_nodes_from(range(N, N+M), bipartite=1)
        
        for i in range(N):
            for j in range(M):
                if np.random.uniform() <= \
                    connection_power_law(distance(thetas[i], phis[j], R) / \
                                         distance_scale(kappas[i], lambdas[j], mu),\
                                             connection_parameter):
                    G.add_edge(i, N+j)
                    
        self.thetas, self.phis, self.kappas, self.lambdas = thetas, phis, kappas, lambdas
        self.graph = G
        
        
        
def experiment_1(fixed_kappa, N, M, connection_parameter,  \
                 lambda_distr = ('power_law', 2.5, 1), Lambda = None, R = 1):
    
    I = np.pi / connection_parameter / np.sin(np.pi / connection_parameter)
    if Lambda != None:
        lambda_bar = Lambda
    else:
        gamma, kappa_0 = lambda_distr[1], lambda_distr[2]
        lambda_bar = (gamma - 1) / (gamma - 2) * kappa_0
    mu = np.pi * R / (M * I *  lambda_bar) 
    expected_mean = mu*M*(np.pi / connection_parameter) * (1/np.sin(np.pi/connection_parameter)) \
        / (np.pi*R) * lambda_bar * fixed_kappa
        
    G = latent_bipartite_graph( N, M, connection_parameter, \
                 lambda_distr = lambda_distr , kappa = fixed_kappa, Lambda = Lambda, R = R, mu = mu)
    
    l = list(G.degree(range(N)))
    degrees = np.array([x[1] for x in l])
    
        
    return (expected_mean, np.mean(degrees))
       
def experiment_2(N = 1000, M = 1000, connection_parameter = 2.5, kappa_distr = ('power_law', 2.5, 3.3), \
                 lambda_distr = ('power_law', 2.5, 3.3), kappa = None, Lambda = None, R = 1):
    
    I = np.pi / connection_parameter / np.sin(np.pi / connection_parameter)
    if Lambda != None:
        lambda_bar = Lambda
    else:
        gamma, kappa_0 = lambda_distr[1], lambda_distr[2]
        lambda_bar = (gamma - 1) / (gamma - 2) * kappa_0
    mu = np.pi * R / (M * I *  lambda_bar) 
    
    G = latent_bipartite_graph( N=N, M=M, connection_parameter=connection_parameter, kappa_distr = kappa_distr, \
        lambda_distr = lambda_distr , kappa = kappa, Lambda = Lambda, R = R, mu = mu)
    
    l1, l2 = list(G.degree(range(N))), list(G.degree(range(N, N+M)))
    degrees_top, degrees_bottom = np.array([x[1] for x in l1]), np.array([x[1] for x in l2])
    
    return (degrees_top, degrees_bottom)
    
    #G_configuration = bipartite.configuration_model(degrees_top, degrees_bottom)

    #l1, l2 = list(G_configuration.degree(range(N))), list(G_configuration.degree(range(N, N+M)))
    #degrees_top_configuration, degrees_bottom_configuration = np.array([x[1] for x in l1]), np.array([x[1] for x in l2])
    
    #G_HH = bipartite.reverse_havel_hakimi_graph(degrees_top, degrees_bottom)

    #l1, l2 = list(G_HH.degree(range(N))), list(G_HH.degree(range(N, N+M)))
    #degrees_top_HH, degrees_bottom_HH = np.array([x[1] for x in l1]), np.array([x[1] for x in l2])
def built_in_fact(x, gamma):
    out = np.zeros(len(x))
    for i in range(len(x)):
        temp = 1
        for j in range(gamma):
            mult = x[i] - j
            if mult <= 1:
                break
            temp *= mult
        out[i] = temp
    return out

    
def comparison_2(degrees, lambda_0 = 3.3, gamma = 2.5, number_samples = 10000):
    x = np.arange(lambda_0, np.exp(6))
    y = (gamma - 1)* (lambda_0**(gamma - 1)) * scipy.special.gamma(x - gamma + 1)/scipy.special.gamma(x+1)
    print(y)
    y = np.nan_to_num(y)
    y = y /sum(y)
    print(y)
    samples = np.random.choice(x, number_samples,
              p=y)
    
    samples = sorted(samples)
    q = 1. * np.arange(len(samples)) / (len(samples) - 1)
    q0 = q[:]
    q = np.log(1 - q)
    samples0 = samples[:]
    samples = np.log(samples)
    
    deg = sorted(degrees)
    p = 1. * np.arange(len(deg)) / (len(deg) - 1)
    p0 = p[:]
    p = np.log(1 - p)
    deg0 = deg[:]
    deg = np.log(deg)
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    ax1.plot(q0, samples0, label = 'approximation')
    ax1.plot(p0,deg0, label = 'true model')
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set(xlabel='', ylabel='node degree')
    ax1.tick_params(axis="x", labelsize=9)
    ax1.tick_params(axis="y", labelsize=9)
    ax1.set_title('Simulated cdf')
    ax1.grid()
    
    ax2.plot(q, samples, label = 'approximation')
    ax2.plot(p,deg, label = 'true model')
    ax2.legend(loc="lower left", fontsize=9)
    ax2.set(xlabel='', ylabel='node degree')
    ax2.tick_params(axis="x", labelsize=9)
    ax2.tick_params(axis="y", labelsize=9)
    ax2.set_title('Linear behaviour - log plot')
    ax2.grid()
    
    return (q, samples, p, deg)



def experiment_3(N = 100, M = 1000, connection_parameter = 2.5, kappa_distr = ('power_law', 2.5, 3.3), \
                 lambda_distr = ('power_law', 2.5, 3.3), kappa = None, Lambda = None, R = 1):
    
    I = np.pi / connection_parameter / np.sin(np.pi / connection_parameter)
    if Lambda != None:
        lambda_bar = Lambda
    else:
        gamma, kappa_0 = lambda_distr[1], lambda_distr[2]
        lambda_bar = (gamma - 1) / (gamma - 2) * kappa_0
    mu = np.pi * R / (M * I *  lambda_bar) 
    
    #prova qui
    R = 0.1
    G = latent_bipartite_graph( N=N, M=M, connection_parameter=connection_parameter, kappa_distr = kappa_distr, \
        lambda_distr = lambda_distr , kappa = kappa, Lambda = Lambda, R = R, mu = mu)
    l1, l2 = list(G.degree(range(N))), list(G.degree(range(N, N+M)))
    degrees_top, degrees_bottom = np.array([x[1] for x in l1]), np.array([x[1] for x in l2])
    
    common_neigh = []
    for i in range(N-1):
        for j in range(i+1, N):
            temp = nx.common_neighbors(G, i, j)
            common_neigh.append(len(list(temp)))
    common_neigh.sort() 
    common_neigh = np.array(common_neigh)      
    clustering = list(bipartite.clustering(G, range(N)).values())
    clustering.sort() 
    clustering = np.array(clustering)        
    
    
    
    
    G_configuration = bipartite.configuration_model(degrees_top, degrees_bottom)
    common_neigh_configuration = []
    for i in range(N-1):
        for j in range(i+1, N):
            temp = nx.common_neighbors(G_configuration, i, j)
            common_neigh_configuration.append(len(list(temp)))
    common_neigh_configuration.sort() 
    common_neigh_configuration = np.array(common_neigh_configuration)      
    clustering_configuration = list(bipartite.clustering(G_configuration, range(N)).values())
    clustering_configuration.sort() 
    clustering_configuration = np.array(clustering_configuration)
    
    G_HH = bipartite.reverse_havel_hakimi_graph(degrees_top, degrees_bottom)
    common_neigh_HH = []
    for i in range(N-1):
        for j in range(i+1, N):
            temp = nx.common_neighbors(G_HH, i, j)
            common_neigh_HH.append(len(list(temp)))
    common_neigh_HH.sort()
    common_neigh_HH = np.array(common_neigh_HH)       
    clustering_HH = list(bipartite.clustering(G_HH, range(N)).values())
    clustering_HH.sort() 
    clustering_HH = np.array(clustering_HH)
    
    neigh = np.stack([common_neigh, common_neigh_configuration, common_neigh_HH], axis = 0)
    cl = np.stack([clustering, clustering_configuration, clustering_HH], axis=0)
    return (neigh, cl)


def experiment_3(N = 100, M = 1000, connection_parameter = 2.5, kappa_distr = ('power_law', 2.5, 3.3), \
                 lambda_distr = ('power_law', 2.5, 3.3), kappa = None, Lambda = None, R = 1):
    
    I = np.pi / connection_parameter / np.sin(np.pi / connection_parameter)
    if Lambda != None:
        lambda_bar = Lambda
    else:
        gamma, kappa_0 = lambda_distr[1], lambda_distr[2]
        lambda_bar = (gamma - 1) / (gamma - 2) * kappa_0
    mu = np.pi * R / (M * I *  lambda_bar) 
    
    #prova qui
    R = 0.1
    G = latent_bipartite_graph( N=N, M=M, connection_parameter=connection_parameter, kappa_distr = kappa_distr, \
        lambda_distr = lambda_distr , kappa = kappa, Lambda = Lambda, R = R, mu = mu)
    l1, l2 = list(G.degree(range(N))), list(G.degree(range(N, N+M)))
    degrees_top, degrees_bottom = np.array([x[1] for x in l1]), np.array([x[1] for x in l2])
    
    common_neigh = []
    for i in range(N-1):
        for j in range(i+1, N):
            temp = nx.common_neighbors(G, i, j)
            common_neigh.append(len(list(temp)))
    common_neigh.sort() 
    common_neigh = np.array(common_neigh)      
    clustering = list(bipartite.clustering(G, range(N)).values())
    clustering.sort() 
    clustering = np.array(clustering)        
    
    
    
    
    G_configuration = bipartite.configuration_model(degrees_top, degrees_bottom)
    common_neigh_configuration = []
    for i in range(N-1):
        for j in range(i+1, N):
            temp = nx.common_neighbors(G_configuration, i, j)
            common_neigh_configuration.append(len(list(temp)))
    common_neigh_configuration.sort() 
    common_neigh_configuration = np.array(common_neigh_configuration)      
    clustering_configuration = list(bipartite.clustering(G_configuration, range(N)).values())
    clustering_configuration.sort() 
    clustering_configuration = np.array(clustering_configuration)
    
    G_HH = bipartite.reverse_havel_hakimi_graph(degrees_top, degrees_bottom)
    common_neigh_HH = []
    for i in range(N-1):
        for j in range(i+1, N):
            temp = nx.common_neighbors(G_HH, i, j)
            common_neigh_HH.append(len(list(temp)))
    common_neigh_HH.sort()
    common_neigh_HH = np.array(common_neigh_HH)       
    clustering_HH = list(bipartite.clustering(G_HH, range(N)).values())
    clustering_HH.sort() 
    clustering_HH = np.array(clustering_HH)
    
    neigh = np.stack([common_neigh, common_neigh_configuration, common_neigh_HH], axis = 0)
    cl = np.stack([clustering, clustering_configuration, clustering_HH], axis=0)
    return (neigh, cl)


def experiment_4(N = 100, M = 1000, connection_parameter = 2.5, kappa_distr = ('power_law', 2.5, 3.3), \
                 lambda_distr = ('power_law', 2.5, 3.3), kappa = None, Lambda = None, R = 1):
    
    I = np.pi / connection_parameter / np.sin(np.pi / connection_parameter)
    if Lambda != None:
        lambda_bar = Lambda
    else:
        gamma, kappa_0 = lambda_distr[1], lambda_distr[2]
        lambda_bar = (gamma - 1) / (gamma - 2) * kappa_0
    mu = np.pi * R / (M * I *  lambda_bar) 
    
    #prova qui
    R = 0.1
    G = latent_bipartite_graph( N=N, M=M, connection_parameter=connection_parameter, kappa_distr = kappa_distr, \
        lambda_distr = lambda_distr , kappa = kappa, Lambda = Lambda, R = R, mu = mu)
    l1, l2 = list(G.degree(range(N))), list(G.degree(range(N, N+M)))
    degrees_top, degrees_bottom = np.array([x[1] for x in l1]), np.array([x[1] for x in l2])
        
    clustering = list(bipartite.clustering(G, range(N)).values())
    clustering.sort() 
    clustering = np.array(clustering)          
    
    G_configuration = bipartite.configuration_model(degrees_top, degrees_bottom)
          
    clustering_configuration = list(bipartite.clustering(G_configuration, range(N)).values())
    clustering_configuration.sort() 
    clustering_configuration = np.array(clustering_configuration)
    
    cl = np.stack([clustering, clustering_configuration], axis=0)
    return cl


def sample_x():
    u = np.random.uniform()
    return np.tan(np.pi*u)

def NMC(n = 250, m = 16, kappa_1 = 0, kappa_2 = np.pi, lambda_bar = 10, lambda_distr = ('power_law', 2.5, 3.3), \
        N= 1000, M=1000, I = np.pi/2, kappa_hat=10, angle_dist=np.pi/2):
    
    out = 0
    angle_dist_tilde = N*I*kappa_hat*angle_dist
    for i in range(n):
        inner = 0
        for j in range(m):
            x = sample_x()
            r_input = kappa_1/kappa_2*x - np.sqrt(kappa_1/kappa_2)*angle_dist_tilde
            inner += connection_power_law(r_input, beta_parameter = 2)
        inner = inner/m
        
        Lambda = sample_power_law(gamma = lambda_distr[1], kappa_0 = lambda_distr[2], sample_size = 1)
        out += Lambda*inner
    out = out/n*kappa_1*np.pi/I/lambda_bar
    return out
        
def binary_search(m_obs, iterations = 8, n = 250, m = 16, kappa_1 = 0, kappa_2 = np.pi, lambda_bar= 10, lambda_distr = ('power_law', 2.5, 3.3), \
         N= 1000, M=1000, I = np.pi/2, kappa_hat=10):
    teta_min, teta_max = 0, np.pi
    for i in range(iterations):
        teta_est = (teta_min + teta_max)/2
        m_est = NMC(n, m , kappa_1 , kappa_2 , lambda_bar, lambda_distr, \
        N, M, I, kappa_hat, teta_est)
        if m_est >= m_obs:
            teta_min = teta_est
        else:
            teta_max = teta_est
    return (teta_min + teta_max)/2

            
            
    
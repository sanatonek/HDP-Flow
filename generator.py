from random import vonmisesvariate
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
import numpy as np
from numpy.random import normal, dirichlet, beta, multinomial, multivariate_normal
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import os
from flows import *
import sys
sns.set();


class HDPHMM:
    def __init__(self, alpha, gma, rho, lamda, n_features, device, n_MADE_blocks=1, periodic=False,
                 MADE_input_size=None, MADE_hidden_size=8, MADE_n_hidden=1, MADE_batch_norm=False):
        """
        Hierarchical non-parametric generative model for time series

        :param lmbda: Parameters of the base distribution H.
            H is a multivariate normal distribution where lmbda[0] is the mean and lmbda[1] indicates the covariance
        :param alpha: Top-level DP parameter
        :param gma: Lower-level DP parameter
        :param rho: Self transition parameter of the Beta distribution
            (kappa[0] is the  logistic function max value and kappa[1] is the logistic growth rate)
        :param lamda: Mean and Standard deviation of the base distribution H
        :param n_features: The size of the multivariate time series data features
        :param nf_model: Normalizing flow model that approximates the data distribution p(x) from the
            distribution of representations p(s)
        """

        self.alpha = alpha
        self.gma = gma
        self.rho = rho
        self.lamda = lamda
        self.k_max = 50
        self.kappa = np.array([beta(rho[0], rho[1]) for _ in range(self.k_max)])
        self.n_features = n_features
        self.device = device
        self.periodic = periodic
        self.MADE_input_size = MADE_input_size
        self.MADE_hidden_size = MADE_hidden_size
        self.MADE_n_hidden = MADE_n_hidden
        self.MADE_batch_norm = MADE_batch_norm
        self.n_MADE_blocks = n_MADE_blocks
        if MADE_input_size is None:
            MADE_input_size = self.n_features

        
        self.flows = WrappedMAF(n_blocks=n_MADE_blocks, # Number of MADE blocks
                                input_size=MADE_input_size, # dimension of distribution you'd like to learn
                                hidden_size=MADE_hidden_size, # dimension of hidden layers in the MADE Blocks
                                n_hidden=MADE_n_hidden, # Number of hidden layers in each MADE block
                                periodic=periodic,
                                cond_label_size=1, # Size of context information we feed at each time step. None if you don't want any.
                                batch_norm=MADE_batch_norm # True iff we want a batch norm layer after each MADE block
                            ).to(self.device)
        s = 0
        self.flows.eval()
        for param in self.flows.parameters():
            s += torch.numel(param) # total number of elements in param
            param.requires_grad = False
        print('Number of parameters in the NF (theta_size): ', s)
        self.theta_size = s
        self.s_size = s

        self.beta = self._gem(gma, self.k_max)
        k_max = len(self.beta)
        self.theta = multivariate_normal(mean=np.zeros(self.theta_size)*lamda[0],
                                         cov=np.diag(np.ones(self.theta_size))*lamda[1], size=k_max)
        self.pi = np.zeros((k_max, k_max))

        for i in range(len(self.beta)):
            eps = self._gem(alpha, 800)
            for j in range(len(eps)):
                phi = np.where(multinomial(1, dirichlet(self.beta)))[0][0]
                self.pi[i][phi] += eps[j]

    def _gem(self, gma, lim=None):
        """
        Stick-breaking process with parameter gma.
        """
        beta_weights = []
        prev = 1
        if lim is None:
            while (prev>0.1 or len(beta_weights)<self.k_min):
                beta_k = beta(1, gma) * prev
                prev -= beta_k
                beta_weights.append(beta_k)
        else:
            while (len(beta_weights) < lim):
                beta_k = beta(1, gma) * prev
                prev -= beta_k
                beta_weights.append(beta_k)
            beta_weights.append(prev)
        return np.array(beta_weights)

    def fill_flow_params(self, theta_vector, nf=None):
        """
        Fills the parameters of self.flows using a theta_vector. 
        theta_vector must be a k dimensional vector, where k is the number of total trainable parameters in
        self.flows

        Set nf to a separate flow if you wish to fill those flow parameters. Else, it'll do it for self.flows
        """
        
        curr_ind = 0
        with torch.no_grad():
            for param in self.flows.parameters():
                size = torch.numel(param) 
                param.copy_(theta_vector[curr_ind: curr_ind + size].reshape(param.shape))
                curr_ind += size
        self.flows.eval()

    def _nf(self, z_t, z_count=None, verbose=False):
        """
        Time series generative process

        z_t is a scalar
        base_dist_params is a tuple of size 2.
          The first element is a torch Tensor of shape (self.n_features,)
          The second element is a torch Tensor of shape (self.n_features,)
        z_count is a scalar. It is the number of steps the time series has been in the current state
        """

        self.fill_flow_params(theta_vector=torch.Tensor(self.theta[z_t]).to(self.device))

        cond = torch.Tensor([z_count]).reshape(1, 1).to(self.device) if z_count is not None else None
        x = self.flows.sample(num_samples=1, context=cond)
        return x[0]

    def log_prob_batch(self, x, theta_list, z=None, cond=None, mini_batch_size=1):
        """
        Takes in data x of shape (n_samples, T, num_features)
        theta_list is of shape (n_samples, k_max, theta_size)
        z is the underlying state of shape (n_samples, T)
        cond is conditioning information of shape (num_samples, T)

        outputs a tensor of shape (num_samples,)
        """

        with torch.no_grad():
            all_probs = []
            for sample_ind, theta_mc in enumerate(theta_list):
                # Looping over all samples
                out = []
                for theta_ind, theta in enumerate(theta_mc):
                    self.fill_flow_params(theta_vector=theta)
                    p = self.flows._log_prob(x[sample_ind],
                                             cond[sample_ind].unsqueeze(-1) if cond is not None else None)
                    out.append(p)
                probs = torch.stack(out, -1)  # [T, k]
                if z is None:
                    all_probs.append(probs)
                else:
                    all_probs.append(probs[np.arange(len(z[sample_ind])), z[sample_ind]])
        return torch.stack(all_probs)

    def generate(self, T, verbose=False):
        """
        Simulate a time series sample
        """
        base_distr_means = []
        base_distr_vars = [] # Store the base distributions for the flow. This changes each time step. 

        sample_x = []
        
        sample_z = list(np.where(multinomial(1, dirichlet(self.beta)))[0])
        x_t = self._nf(z_t=sample_z[0], z_count=torch.Tensor([1]), verbose=verbose)
        x_t = x_t.cpu().detach().numpy()
        if verbose:
            print('x_t: ', x_t)
        sample_x.append(x_t) 
        z_count = 1
        for _ in range(1, T):
            if verbose:
                print('Time Step: ', _)
            self_trans = np.diag(self.kappa)
            z_t = np.where(multinomial(1, dirichlet(self.pi[sample_z[-1], :] + self_trans[sample_z[-1]] + 1e-22)))[0][0]
            if z_t == sample_z[-1]:
                z_count += 1
            else:
                z_count = 1
            p_s = torch.distributions.MultivariateNormal(loc=torch.Tensor(self.theta[z_t]),
                                                         scale_tril=torch.diag(torch.ones(len(self.theta[z_t]))*0.5))
            
            x_t = self._nf(z_t=sample_z[-1], z_count=torch.Tensor([z_count]), verbose=verbose)
           
            
            x_t = x_t.cpu().detach().numpy()
            sample_x.append(x_t)
            sample_z.append(z_t)
        return np.array(sample_z), np.stack(sample_x, axis=0)


if __name__=="__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N = 650
    n_features = 4
    T = 40
    k_max = 10
    model = HDPHMM(alpha=30, gma=8, kappa=[8,1], k_min=k_max, n_features=n_features, theta_size=30, device=device)
    x_all, z_all = [], []
    for _ in range(N):
        z, x = model.generate(T, verbose=False)
        x_all.append(x)
        z_all.append(z)
    x_all = np.array(x_all)
    z_all = np.array(z_all)
    with open('./data/sim_hard_x.npy', 'wb') as f:
        np.save(f, x_all)
    with open('./data/sim_hard_z.npy', 'wb') as f:
        np.save(f, z_all)

    if not os.path.exists('./plots'):
        os.makedirs('./plots')

    fig, axs = plt.subplots(8, 1, figsize=(20, 8))
    theta_space = ["S%d" % i for i in range(model.k_min)]
    sns.barplot(x=theta_space, y=model.beta, ax=axs[0])
    axs[0].set_ylabel("G")
    axs[0].set_ylim(0, 0.5)
    for i in range(1, 8):
        sns.barplot(x=theta_space, y=model.pi[i - 1], ax=axs[i])
        axs[i].set_ylabel("G%d" % (i - 1))
        axs[i].set_ylim(0, 0.5)
    plt.tight_layout()
    plt.savefig("./plots/dp_distributions.pdf")

    plt.figure(figsize=(10, 3))
    plt.xticks(ticks=np.array([10*i for i in range(T//10)]), labels = np.array([10*i for i in range(T//10)]))
    for i in range(x.shape[-1]):
        plt.plot(x[:,i], label="Feature %d"%i)

    color = iter(plt.cm.rainbow(np.linspace(0, 1, 10)))
    z_colors={}
    for t in range(T-1):
        if not z[t] in z_colors.keys():
            z_colors[z[t]] = next(color)
        plt.axvspan(t, t+1, alpha=0.5, color=z_colors[z[t]])
    plt.title("Time series sample with underlying states")
    plt.savefig("./plots/ts_sample.pdf")


    plt.figure()
    N = 50
    s_dist = []
    z_dist = []
    for _ in range(N):
        s, z, _ = model.generate(50)
        s_dist.append(s)
        z_dist.append(z)
    s_dist = np.array(s_dist)
    z_dist = np.array(z_dist)
    sns.scatterplot(x=s_dist[:,:,0].reshape(-1,), y=s_dist[:,:,1].reshape(-1,), hue=z_dist.reshape(-1,))
    plt.title("Distribution of representations (S) of %d samples of length %d"%(N, T))
    plt.savefig("./plots/representation_distribution.pdf")

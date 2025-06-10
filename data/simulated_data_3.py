from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
from numpy.random import dirichlet, multinomial, multivariate_normal
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
import os
from flows import *
sns.set();


class HDPHMM:
    def __init__(self, device):
        """
        Generative model with fixed parameters to create a dataset
        """
        # self.theta_size = 231
        # self.kappa = np.array([0.65, 0.6, 0.7, 0.75, 0.55, 0.5])
        self.flows = WrappedMAF(n_blocks=1,  # Number of MADE blocks
                                input_size=4,  # dimension of distribution you'd like to learn
                                hidden_size=6,  # dimension of hidden layers in the MADE Blocks
                                n_hidden=1,  # Number of hidden layers in each MADE block
                                cond_label_size=1,
                                # Size of context information we feed at each time step. None if you don't want any.
                                batch_norm=True  # True iff we want a batch norm layer after each MADE block
                                )

        s = 0
        self.flows.eval()
        for param in self.flows.parameters():
            s += torch.numel(param)  # total number of elements in param
        print('num params for nf: ', s)
        self.theta_size = s

        self.k_max = 6
        self.n_features = 4
        self.device = device

        self.beta = np.array([0.4, 0.2, 0.12, 0.14, 0.04, 0.1])
        k_max = len(self.beta)
        self.theta = multivariate_normal(mean=np.zeros(self.theta_size),
                                         cov=np.diag(np.ones(self.theta_size)),
                                         size=k_max)
        self.pi = np.array([[0.7, 0.1, 0.06, 0.07, 0.01, 0.06],
                            [0.18, 0.67, 0.05, 0.05, 0.00, 0.05],
                            [0.15, 0.07, 0.66, 0.06, 0.02, 0.03],
                            [0.11, 0.08, 0.07, 0.67, 0.02, 0.05],
                            [0.10, 0.09, 0.06, 0.08, 0.63, 0.04],
                            [0.11, 0.07, 0.07, 0.06, 0.02, 0.67]])

        # self.flows = MAF(n_blocks=1,  # Number of MADE blocks
        #                  input_size=self.n_features,  # dimension of distribution you'd like to learn
        #                  hidden_size=self.n_features // 2,  # dimension of hidden layers in the MADE Blocks
        #                  n_hidden=1,  # Number of hidden layers in each MADE block
        #                  cond_label_size=1,
        #                  # Size of context information we feed at each time step. None if you don't want any.
        #                  batch_norm=False  # True iff we want a batch norm layer after each MADE block
        #                  ).to(self.device)

    def fill_flow_params(self, theta_vector, nf=None):
        """
        Fills the parameters of self.flows using a theta_vector.
        theta_vector must be a k dimensional vector, where k is the number of total trainable parameters in
        self.flows
        """
        curr_ind = 0
        with torch.no_grad():
            for param in self.flows.parameters():
                size = torch.numel(param)  # total number of elements in param
                param.copy_(theta_vector[curr_ind: curr_ind + size].reshape(param.shape))
                curr_ind += size

    def _nf(self, z_t, z_count=None):
        """
        Time series generative process
        """
        self.fill_flow_params(theta_vector=torch.Tensor(self.theta[z_t]).to(self.device))

        cond = torch.Tensor([z_count]).reshape(1, 1).to(self.device) if z_count is not None else None
        x = self.flows.sample(num_samples=1, context=cond)
        return x[0]

    def generate(self, T):
        """
        Simulate a time series sample
        """
        sample_x = []
        sample_z = list(np.where(multinomial(1, dirichlet(self.beta)))[0])
        x_t = self._nf(z_t=sample_z[0], z_count=torch.Tensor([1]))
        x_t = x_t.cpu().detach().numpy()
        sample_x.append(x_t)
        z_count = 1
        for _ in range(1, T):
            # Estimate the underlying state and encourage self-transition
            # self_trans = np.zeros_like(self.pi[sample_z[-1], :])
            # self_trans[sample_z[-1]] += self.kappa[0]/(1+np.exp(-self.kappa[1]*z_count))
            # self_trans = np.diag(self.kappa)
            self_trans = self.pi# * (1 - self.kappa)[..., np.newaxis] + np.diag(self.kappa)
            z_t = np.where(multinomial(1, self_trans[sample_z[-1]]))[0][0]
            # z_t = np.where(multinomial(1, dirichlet(self.pi[sample_z[-1], :] + self_trans[sample_z[-1]] + 1e-22)))[0][0]
            if z_t == sample_z[-1]:
                z_count += 1
            else:
                z_count = 1
            x_t = self._nf(z_t=sample_z[-1], z_count=torch.Tensor([z_count]))

            x_t = x_t.cpu().detach().numpy()
            sample_x.append(x_t)
            sample_z.append(z_t)
        return np.array(sample_z), np.concatenate(sample_x)


    # def fill_flow_params(self, theta_vector, nf=None):
    #     """
    #     Fills the parameters of self.flows using a theta_vector.
    #     theta_vector must be a k dimensional vector, where k is the number of total trainable parameters in
    #     self.flows
    #
    #     Set nf to a separate flow if you wish to fill those flow parameters. Else, it'll do it for self.flows
    #     """
    #     flow_model = nf if nf is not None else self.flows
    #
    #     curr_ind = 0
    #     with torch.no_grad():
    #         for param in flow_model.parameters():
    #             size = torch.numel(param)  # total number of elements in param
    #             param.copy_(theta_vector[curr_ind: curr_ind + size].reshape(param.shape))
    #             curr_ind += size
    #
    # def _nf(self, z_t, base_dist_params, z_count=None, verbose=False):
    #     self.fill_flow_params(theta_vector=torch.Tensor(self.theta[z_t]).to(self.device))
    #     # THIS IS THE LINE THAT UPDATES THE BASE DIST EACH TIME STEP
    #     u = self.flows.net[0].base_dist.sample().to(self.device)
    #
    #     cond = torch.Tensor([z_count]).reshape(1, 1).to(self.device) if z_count is not None else None
    #     x, sum_log_abs_det_jacobians, list_dist_params = self.flows.inverse(u=u, cond=cond, return_dist_params=True)
    #
    #     # logsigma is log of sigma. To get sigma, simply apply exp
    #     mu, logsigma = list_dist_params[0]  # NOTE This line will need to change if we have multiple flows i.e. multiple MADE blocks
    #     # mu and loglogitsigma are of shape (1, self.n_features), move to cpu
    #     mu, logsigma = mu.reshape(-1, ).cpu(), logsigma.reshape(-1, ).cpu()
    #
    #     if verbose:
    #         print('mu: ', mu)
    #         print('logsigma: ', logsigma)
    #         print('torch.exp(logsigma): ', torch.exp(logsigma))
    #     return x, MultivariateNormal(loc=mu.to(self.device),
    #                                  scale_tril=(torch.diag(torch.exp(logsigma)) + torch.eye(self.n_features) * 1e-5))
    #
    # def generate(self, T, verbose=False):
    #     """
    #     Simulate a time series sample
    #     """
    #     sample_x = []
    #     sample_z = list(np.where(multinomial(1, dirichlet(self.beta)))[0])
    #     x_t, p_x = self._nf(z_t=sample_z[0],
    #                         base_dist_params=(torch.zeros(self.n_features), torch.ones(self.n_features)),
    #                         z_count=torch.Tensor([1]), verbose=verbose)
    #     x_t = x_t.cpu().detach().numpy()
    #     sample_x.append(x_t)
    #     z_count = 1
    #     self_trans = np.diag(self.kappa)
    #     transition_matrix = self.pi * (1-self.kappa)[:,np.newaxis] + self_trans
    #     for _ in range(1, T):
    #         z_t = np.where(multinomial(1, dirichlet(transition_matrix[sample_z[-1]] + 1e-22)))[0][0]
    #         if z_t == sample_z[-1]:
    #             z_count += 1
    #         else:
    #             z_count = 1
    #         x_t, p_x = self._nf(z_t=sample_z[-1], base_dist_params=(p_x.loc, torch.diagonal(p_x.scale_tril)),
    #                             z_count=torch.Tensor([z_count]), verbose=verbose)
    #         x_t = x_t.cpu().detach().numpy()
    #         sample_x.append(x_t)
    #         sample_z.append(z_t)
    #     return np.array(sample_z), np.stack(sample_x, axis=0)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    np.random.seed(1234)
    N = 300
    n_features = 4
    T = 50
    k_max = 10
    model = HDPHMM(device=device)
    x_all, z_all = [], []
    for _ in range(N):
        z, x = model.generate(T)
        x_all.append(x)
        z_all.append(z)
    x_all = np.array(x_all)
    z_all = np.array(z_all)
    with open('./sim_hard_x_2.npy', 'wb') as f:
        np.save(f, x_all)
    with open('./sim_hard_z_2.npy', 'wb') as f:
        np.save(f, z_all)

    if not os.path.exists('./plots'):
        os.makedirs('./plots')

    plt.figure(figsize=(10, 3))
    plt.xticks(ticks=np.array([10 * i for i in range(T // 10)]), labels=np.array([10 * i for i in range(T // 10)]))
    for i in range(x.shape[-1]):
        plt.plot(x[:, i], label="Feature %d" % i)

    color = iter(plt.cm.rainbow(np.linspace(0, 1, 10)))
    z_colors = {}
    for t in range(T - 1):
        if not z[t] in z_colors.keys():
            z_colors[z[t]] = next(color)
        plt.axvspan(t, t + 1, alpha=0.5, color=z_colors[z[t]])
    plt.title("Time series sample with underlying states")
    plt.savefig("./plots/ts_sample.pdf")

    # fig, axs = plt.subplots(6, 1, figsize=(20, 6))
    # theta_space = ["S%d" % i for i in range(model.k_max)]
    # sns.barplot(x=theta_space, y=model.beta, ax=axs[0])
    # axs[0].set_ylabel("G")
    # axs[0].set_ylim(0, 0.5)
    # for i in range(1, 6):
    #     sns.barplot(x=theta_space, y=model.pi[i - 1], ax=axs[i])
    #     axs[i].set_ylabel("G%d" % (i - 1))
    #     axs[i].set_ylim(0, 0.5)
    # plt.tight_layout()
    # plt.savefig("./plots/dp_distributions.pdf")

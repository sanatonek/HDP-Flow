import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import torch

from torch.distributions import Categorical, Normal, Dirichlet, Beta as Beta_torch

sns.set()
sys.path.append('..')
torch.autograd.set_detect_anomaly(True)

class HDP_FLOW(torch.nn.Module):
    def __init__(self, data, prior_model, k_max, device):
        super(HDP_FLOW, self).__init__()
        self.k_max = k_max
        self.param_size = prior_model.theta_size
        self.prior_model = prior_model
        self.prior_model.flows.eval()
        self.data = torch.Tensor(data).to(device)
        self.N = len(data)
        self.T, self.dim_size = data[0].shape
        self.theta_size = prior_model.theta_size
        self.device = device

        self.pi_param = torch.nn.Parameter(torch.zeros(self.k_max, self.k_max, device=device))
        self.beta_param = torch.nn.Parameter(torch.zeros(self.k_max, device=device))
        self.theta_mu = torch.nn.Parameter(torch.zeros(self.k_max, self.theta_size, device=device))
        self.z_params = torch.nn.Parameter(torch.zeros((self.N, self.T, self.k_max), device=device))
        self.kappa_params = torch.nn.Parameter(torch.ones((self.k_max, 2), device=device) * 2)

        self.G_pi = torch.zeros_like(self.pi_param).to(device)
        self.M_pi = torch.zeros_like(self.pi_param).to(device)
        self.G_beta = torch.zeros_like(self.beta_param).to(device)
        self.M_beta = torch.zeros_like(self.beta_param).to(device)
        self.G_theta = torch.zeros_like(self.theta_mu).to(device)
        self.M_theta = torch.zeros_like(self.theta_mu).to(device)
        self.G_z = torch.zeros_like(self.z_params).to(device)
        self.M_z = torch.zeros_like(self.z_params).to(device)
        self.G_kappa = torch.zeros_like(self.kappa_params).to(device)
        self.M_kappa = torch.zeros_like(self.kappa_params).to(device)

    def _get_q_dist(self, ind=None):
        if ind is None:
            q_z = Categorical(logits=self.z_params)
        else:
            q_z = Categorical(logits=self.z_params[ind])
        q_pi = Dirichlet(torch.nn.Softplus()(self.pi_param) + 1e-10)
        q_theta = Normal(loc=self.theta_mu, scale=torch.ones_like(self.theta_mu) * 0.2)
        q_beta = Dirichlet(torch.nn.Softplus()(self.beta_param) + 1e-10)
        q_kappa = Beta_torch(self.kappa_params[:, 0] + 1e-5, self.kappa_params[:, 1] + 1e-5)
        return q_z, q_pi, q_theta, q_beta, q_kappa

    def sample_q(self, ind=None, n_samples=100):
        q_z, q_pi, q_theta, q_beta, q_kappa = self._get_q_dist(ind)
        z = q_z.sample((n_samples,))
        theta = q_theta.sample((n_samples,))
        pi = q_pi.sample((n_samples,))
        beta = q_beta.sample((n_samples,))
        kappa = q_kappa.sample((n_samples,))
        return z, theta, pi, beta, kappa

    def _clear_grads(self):
        for param in self.parameters():
            if param.grad is not None:
                param.grad.data.zero_()

    def logq_var(self, latents, ind):
        q_z, q_pi, q_theta, q_beta, q_kappa = self._get_q_dist(ind)
        z, theta, pi, beta, kappa = latents
        logq_theta = q_theta.log_prob(theta)
        logq_pi = q_pi.log_prob(pi)
        logq_beta = q_beta.log_prob(beta)
        logq_z = q_z.log_prob(z)
        logq_kappa = q_kappa.log_prob(kappa)
        return logq_z, logq_theta.sum(-1), logq_pi, logq_kappa, logq_beta

    def logp_latent(self, latents):
        z, theta, pi, beta, kappa = latents
        l_p = 100
        ext = torch.zeros((len(beta), l_p - self.k_max), device=self.device) + 1e-10
        p_beta = Dirichlet(torch.full((l_p,), self.prior_model.gma / l_p, device=self.device) + 1e-5)
        logp_beta = p_beta.log_prob(torch.cat((beta, ext), -1))

        trans_prob = pi
        p_z_init = Categorical(probs=beta)
        logp_z = [p_z_init.log_prob(z[:, 0])]
        logp_z_pc = torch.zeros((len(z), self.k_max, z.shape[-1]), device=self.device)
        for t in range(1, z.shape[-1]):
            p_zt = Categorical(trans_prob[np.arange(len(z)), z[:, t - 1]])
            logp_z_pc[np.arange(z.shape[0]), z[:, t - 1], t] = p_zt.log_prob(z[:, t])
        logp_z_pc[:, :, 0] = (logp_z[0] / self.k_max).unsqueeze(-1).repeat((1, self.k_max))

        mean = torch.ones((self.k_max, self.theta_size), device=self.device)
        p_theta = Normal(loc=mean * self.prior_model.lamda[0], scale=torch.ones_like(mean) * self.prior_model.lamda[1])
        logp_theta = p_theta.log_prob(theta).sum(-1)

        p_kappa = Beta_torch(
            torch.tensor(self.prior_model.rho[0], device=self.device),
            torch.tensor(self.prior_model.rho[1], device=self.device)
        )
        logp_kappa = p_kappa.log_prob(kappa)

        logp_pi = torch.zeros((len(pi),), device=self.device)
        for k in range(self.k_max):
            delt = torch.zeros((l_p,), device=self.device)
            delt[k] = 1
            p_pi = Dirichlet(
                self.prior_model.alpha * torch.cat((beta, ext), -1) + delt * torch.cat((kappa, ext), -1) + 1e-5
            )
            logp_pi += p_pi.log_prob(torch.cat((pi[:, k], ext), -1))

        return logp_z_pc, logp_theta, logp_pi, logp_kappa, logp_beta

    def logp_x_cond(self, latents, x):
        z, theta, _, _, _ = latents
        _, T = x.shape[:2]
        n_samples = len(z)
        z = z[:, :T]
        x = x.repeat((n_samples, 1, 1))
        logp_x_pc = torch.zeros((n_samples, self.k_max, T)).to(self.device)
        z_count = torch.ones(n_samples).to(self.device)
        z_counts = torch.ones(n_samples, T).to(self.device)
        for t in range(1, T - 1):
            z_count = torch.where(z[:, t] == z[:, t - 1], z_count + 1, torch.ones(n_samples).to(self.device))
            z_counts[:, t] = z_count
        logp_x = self.prior_model.log_prob_batch(x, theta, z.to(self.device), cond=z_counts)
        for t in range(T):
            logp_x_pc[np.arange(n_samples), z[:, t], t] = logp_x[:, t]
        return logp_x_pc

  def grad_step(self, x, lr, n_samples, ind, optim='Adam', use_control_variates=True, print_res=False):
        for i in range(1):
            latents = self.sample_q(n_samples=n_samples, ind=ind)
            logq_z_all, logq_theta_all, logq_pi_all, logq_kappa_all, logq_beta_all = self.logq_var(latents, ind=ind)
            logp_z_all, logp_theta_all, logp_pi_all, logp_kappa_all, logp_beta_all = self.logp_latent(latents)
            logpx_z_all = self.logp_x_cond(latents, x)

            # Grad estimation with control variates
            T = x.shape[-2]
            h_beta_grads, h_theta_grads, h_pi_grads, h_kappa_grads = [], [], [], []
            h_z_grads = []
            elbo_est, elbo_local, elbo_global = [], [], []
            elbo_theta, elbo_pi, elbo_kappa, elbo_beta = [], [], [], []
            for n in range(n_samples):
                logq_z, logq_theta, logq_pi, logq_kappa, logq_beta = logq_z_all[n], logq_theta_all[n], logq_pi_all[n], logq_kappa_all[n], logq_beta_all[n]
                logp_z, logp_theta, logp_pi, logp_kappa, logp_beta = logp_z_all[n], logp_theta_all[n], logp_pi_all[n], logp_kappa_all[n], logp_beta_all[n]
                logpx_z = logpx_z_all[n]
                logp_local = logp_z[:T].sum()
                logq_local = logq_z[:T].sum()
                logp_global = logp_theta.sum() + logp_pi.sum() + logp_kappa.sum() + logp_beta
                logq_global = logq_theta.sum() + logq_pi.sum() + logq_kappa.sum() + logq_beta
                logp = (self.N/len(x))*(logp_local + logpx_z.sum()) + logp_global
                logq = (self.N/len(x))*logq_local + logq_global
                elbo_est.append(logp - logq)

                (logq_global+logq_local).backward(retain_graph=True)
                h_pi_grads.append(self.pi_param.grad.detach().clone())
                h_theta_grads.append(self.theta_mu.grad.detach().clone())
                h_beta_grads.append(self.beta_param.grad.detach().clone())
                h_kappa_grads.append(self.kappa_params.grad.detach().clone())

                # Global update
                elbo_beta.append(logp_beta + logp_pi.sum() - logq_beta)
                elbo_kappa.append((logp_kappa - logq_kappa) + logp_pi)
                elbo_theta.append((logp_theta - logq_theta) + logpx_z.sum(-1))
                elbo_pi.append((logp_pi - logq_pi) + logp_z[:, :T].sum(-1))

                # Local update
                h_z_grads.append(self.z_params.grad.detach().clone())
                h_z_grads[-1][ind, T:, :] = 0
                elbo_local_all = torch.zeros(self.z_params.shape[0], self.z_params.shape[1]).to(self.device)
                elbo_local_all[ind, :T] = (logp_z[:, :T].sum(0) + logpx_z.sum(0) - logq_z[:T])
                elbo_local.append(elbo_local_all)
                self._clear_grads()

            elbo_local = torch.stack(elbo_local)
            elbo_pi = torch.stack(elbo_pi)
            elbo_beta = torch.stack(elbo_beta)
            elbo_theta = torch.stack(elbo_theta)
            elbo_kappa = torch.stack(elbo_kappa)
            elbo_est = torch.stack(elbo_est)

            h_z_grads = torch.stack(h_z_grads)
            h_beta_grads = torch.stack(h_beta_grads)
            h_theta_grads = torch.stack(h_theta_grads)
            h_kappa_grads = torch.stack(h_kappa_grads)
            h_pi_grads = torch.stack(h_pi_grads)

            f_pi_grads = h_pi_grads * (elbo_pi).unsqueeze(-1)#.unsqueeze(-1)
            f_beta_grads = h_beta_grads * (elbo_beta).unsqueeze(-1)
            f_theta_grads = h_theta_grads * (elbo_theta).unsqueeze(-1)#.unsqueeze(-1)
            f_kappa_grads = h_kappa_grads * (elbo_kappa).unsqueeze(-1)#.unsqueeze(-1)
            f_z_grads = h_z_grads * (elbo_local).unsqueeze(-1)#.unsqueeze(-1)

            if use_control_variates:
                grads, weights = [], []
                for list_ind, lmbda in enumerate(zip([f_z_grads, f_beta_grads, f_pi_grads, f_theta_grads, f_kappa_grads],
                                                     [h_z_grads, h_beta_grads, h_pi_grads, h_theta_grads, h_kappa_grads])):
                    f_i, h_i = lmbda
                    if list_ind == 0:  # for Z
                        if ind is not None:
                            f_i = f_i[:, ind]
                            h_i = h_i[:, ind]
                    N = len(f_i)
                    f_i_flat = f_i.reshape((len(f_i), -1)).detach()  # Shape=[n_samples, n_i]
                    h_i_flat = h_i.reshape((len(h_i), -1)).detach()  # Shape=[n_samples, n_i]
                    num = torch.sum(torch.bmm((f_i_flat - torch.mean(f_i_flat, keepdim=True, axis=0)).T.unsqueeze(1),
                                              (h_i_flat - torch.mean(h_i_flat, keepdim=True, axis=0)).T.unsqueeze(-1)) / (
                                                N - 1))
                    denom = torch.sum(torch.bmm((h_i_flat - torch.mean(h_i_flat, keepdim=True, axis=0)).T.unsqueeze(1),
                                                (h_i_flat - torch.mean(h_i_flat, keepdim=True, axis=0)).T.unsqueeze(-1)) / (
                                                  N - 1))
                    a = num / denom
                    a = torch.nan_to_num(a)
                    weights.append(a)
                z_grads = (f_z_grads - (weights[0] * h_z_grads)).mean(0)
                beta_grads = (f_beta_grads - (weights[1] * h_beta_grads)).mean(0)
                pi_grads = (f_pi_grads - (weights[2] * h_pi_grads)).mean(0)
                theta_grads = (f_theta_grads - (weights[3] * h_theta_grads)).mean(0)
                kappa_grads = (f_kappa_grads - (weights[4] * h_kappa_grads)).mean(0)
            else:
                z_grads = f_z_grads.mean(0)
                beta_grads = f_beta_grads.mean(0)
                pi_grads = f_pi_grads.mean(0)
                theta_grads = f_theta_grads.mean(0)
                kappa_grads = f_kappa_grads.mean(0)

            with torch.no_grad():
                if optim == 'Adam':
                    beta_2 = 0.999
                    beta_1 = 0.9
                    self.G_z[ind] = self.G_z[ind]*beta_2 + (z_grads[ind] ** 2)*(1-beta_2)
                    self.M_z[ind] = self.M_z[ind]*beta_1 + (z_grads[ind])*(1-beta_1)
                    self.G_pi = self.G_pi*beta_2 + (pi_grads ** 2)*(1-beta_2)
                    self.M_pi = self.M_pi*beta_1 + (pi_grads)*(1-beta_1)
                    self.G_beta = self.G_beta*beta_2 + (beta_grads ** 2)*(1-beta_2)
                    self.M_beta = self.M_beta*beta_1 + (beta_grads)*(1-beta_1)
                    self.G_kappa = self.G_kappa*beta_2 + (kappa_grads ** 2)*(1-beta_2)
                    self.M_kappa = self.M_kappa*beta_1 + (kappa_grads)*(1-beta_1)
                    self.G_theta = self.G_theta*beta_2 + (theta_grads ** 2)*(1-beta_2)
                    self.M_theta = self.M_theta*beta_1 + (theta_grads)*(1-beta_1)
                    lr_z = torch.zeros_like(self.z_params).to(self.device)
                    lr_z[ind] = lr / (torch.sqrt(self.G_z[ind] + 1e-10))
                    lr_pi = lr / (torch.sqrt(self.G_pi + 1e-10))
                    lr_beta = lr / (torch.sqrt(self.G_beta + 1e-10))
                    lr_kappa = lr / (torch.sqrt(self.G_kappa + 1e-10))
                    lr_theta = lr / (torch.sqrt(self.G_theta + 1e-10))
                    self.z_params.copy_(self.z_params + lr_z * self.M_z)
                    self.pi_param.copy_(self.pi_param + lr_pi * self.M_pi)
                    self.theta_mu.copy_(self.theta_mu + lr_theta * self.M_theta)
                    self.kappa_params.copy_(torch.clamp(self.kappa_params + lr_kappa * self.M_kappa, min=0.1))
                    self.beta_param.copy_(self.beta_param + lr_beta * self.M_beta)

                elif optim=='AdaGrad':
                    self.G_z[ind] = self.G_z[ind] + (z_grads[ind] ** 2)
                    lr_z = torch.zeros_like(self.z_params).to(self.device)
                    lr_z[ind] = lr / (torch.sqrt(self.G_z[ind] + 1e-10))
                    self.G_pi = self.G_pi + (pi_grads ** 2)
                    lr_pi = lr / (torch.sqrt(self.G_pi + 1e-10))
                    self.G_beta = self.G_beta + (beta_grads ** 2)
                    lr_beta = lr / (torch.sqrt(self.G_beta + 1e-10))
                    self.G_kappa = self.G_kappa + (kappa_grads ** 2)
                    lr_kappa = lr / (torch.sqrt(self.G_kappa + 1e-10))
                    self.G_theta = self.G_theta + (theta_grads ** 2)
                    lr_theta = lr / (torch.sqrt(self.G_theta + 1e-10))
                    self.z_params.copy_(self.z_params + (lr_z * z_grads))
                    self.pi_param.copy_(self.pi_param + (lr_pi * pi_grads))
                    self.theta_mu.copy_(self.theta_mu + torch.clip(lr_theta * self.M_theta, min=-0.1, max=+0.1))
                    self.kappa_params.copy_(self.kappa_params + (lr_kappa * kappa_grads))
                    self.beta_param.copy_(self.beta_param + (lr_beta * beta_grads))

                else:
                    raise RuntimeError('Optimizer not implemented!')
        if print_res:
            print("\t\t ELBO: %.3f" % torch.mean(elbo_est))
            print("\t\t Logp(x): %.3f" % torch.mean(logp).item(),
                  "(%.3f, %.3f, %.3f)" % (torch.mean(logp_local).item(), torch.mean(logp_global).item(), torch.mean(logpx_z.sum(-1).sum(-1)).item()))
            print("\t\t Logq(z): %.3f, %.3f" %(torch.mean(logq_local).item(), torch.mean(logq_global).item()))
        return torch.mean(elbo_est).item(), torch.mean(logp).item()

    def _clear_grads(self):
        for param in self.parameters():
            if param.grad is not None:
                param.grad.data.zero_()

    def train(self, X, X_lens, n_epochs, n_mc_sample, lr_init, data, mbatch_size=5, decay=0.01, ckpt_name="vi"):
        epoch_elbo, epoch_logx, epoch_pp = [], [], []
        print('Learning rate: ', lr_init)
        best_elbo = -1*np.inf
        for epoch in range(n_epochs):
            batch_elbo, batch_logx = [], []
            shuffled_ind = np.arange(0, len(X))
            np.random.shuffle(shuffled_ind)
            for batch_ind in shuffled_ind:
                x_batch = torch.Tensor(X[batch_ind, :X_lens[batch_ind]]).unsqueeze(0).to(self.device)
                elbo, logp_x = self.grad_step(x_batch, n_samples=n_mc_sample, lr=lr_init, optim='Adam',
                                              ind=batch_ind, use_control_variates=True, print_res=False)
                batch_elbo.append(elbo)
                batch_logx.append(logp_x)
            if np.mean(batch_elbo) > best_elbo:
                torch.save(self, "./ckpt/%s.pt" %(ckpt_name))
                best_elbo = np.mean(batch_elbo)
            epoch_elbo.extend((batch_elbo))
            epoch_logx.append(np.mean(batch_logx))
            print("=====> Epoch %d \t\t ELBO: %.3f \t\t logp(x,z): %.3f" % (epoch, np.mean(epoch_elbo[-1*len(X):]), epoch_logx[-1]))

            # Plot prior dist
            _, q_pi, _, q_beta, q_kappa = self._get_q_dist()
            init_probs = q_beta.sample((100,)).mean(0)
            colors = list(sns.color_palette('Set3', self.k_max + 1))
            plt.pie(init_probs.cpu().detach().numpy(), colors=colors)
            plt.savefig("./plots/%s/state_dist_e%d.pdf" % (data, epoch))
        print('*********** Best Elbo: ', best_elbo)
        plt.figure()
        plt.plot(np.array(epoch_elbo))
        plt.title("ELBO estimate")
        plt.savefig("./plots/%s/training_elbo_%s.pdf" %(data, ckpt_name))
        plt.close()
        plt.figure()

    def find_state(self, X, lens=None, n_mc_sample=20):
        if lens is None:
            lens = torch.Tensor([X.shape[1]] * len(X)).to(self.device)
        _, pi_dist, theta_dist, beta_dist, _ = self._get_q_dist()
        gamma, posterior_like = [], []
        for n_mc in range(n_mc_sample):
            init_probs = beta_dist.sample()
            pi, theta = pi_dist.sample(), theta_dist.sample()
            pz_X, px_X = self.forward_backward(obs_seq=X, A=pi, init_probs=init_probs, theta=theta, lens=lens)
            gamma.append(pz_X)
            posterior_like.append(px_X)
        gamma = torch.stack(gamma).mean(0)
        log_like = torch.logsumexp(torch.stack(posterior_like), axis=0) - torch.log(torch.Tensor([n_mc_sample]*len(X)))
        return gamma, torch.argmax(gamma, axis=-1), log_like.numpy()

    def posterior_predictive(self, x_hat, n_mc_sample=20, x_hat_lens=None):
        if x_hat_lens is None:
            x_hat_lens = torch.Tensor([x_hat.shape[1]] * len(x_hat)).to(self.device)
        N, T, _ = x_hat.shape
        _, pi_dist, theta_dist, beta_dist, _ = self._get_q_dist()
        posterior_like = []
        for n_mc in range(n_mc_sample):
            init_probs = beta_dist.sample()
            pi, theta = pi_dist.sample(), theta_dist.sample()
            px_X = self.forward_backward(x_hat, pi, theta, init_probs=init_probs, lens=x_hat_lens, forward_only=True)
            posterior_like.append(px_X)
        log_like = torch.logsumexp(torch.stack(posterior_like), axis=0) - torch.log(torch.Tensor([n_mc_sample]*len(x_hat)))
        return log_like.mean(0).item()

    def forward_backward(self, obs_seq, A, theta, init_probs, lens, forward_only=False):
        N, T, F = obs_seq.shape
        obs_seq = obs_seq.to(self.device)
        ## Forward
        z_count = torch.ones((N,)).to(self.device)
        alpha = torch.zeros((N, T, self.k_max)).to(self.device)
        logp_x_zt_new = self.prior_model.log_prob_batch(obs_seq, theta.unsqueeze(0).repeat(N, 1, 1), z=None,
                                                        cond=torch.ones((N, T)).to(self.device).float())  # [N,T,k]
        logp_x_zt_two = self.prior_model.log_prob_batch(obs_seq, theta.unsqueeze(0).repeat(N, 1, 1), z=None,
                                                        cond=torch.ones((N, T)).to(self.device).float()+1)  # [N,T,k]
        logp_x = [logp_x_zt_new[:, 0, :]]
        alpha[:, 0, :] = (torch.exp((logp_x[-1]))+1e-25) * torch.stack([init_probs] * len(obs_seq)).to(self.device)
        cs = torch.log(torch.sum(alpha[:, 0, :], -1))
        alpha[:, 0, :] = alpha[:, 0, :] / torch.sum(alpha[:, 0, :], -1, keepdim=True)
        log_like = torch.zeros(N, ).to(self.device)
        for t in range(1, T):
            # Logp if state didn't change at t
            logp_x_zt_cont = self.prior_model.log_prob_batch(obs_seq[:, t, :].unsqueeze(0), theta.unsqueeze(0),
                                                             z=None, cond=(z_count + 1.).unsqueeze(0))[0]
            p_x_mat = torch.exp(logp_x_zt_new[:, t, :].unsqueeze(1).repeat(1, self.k_max, 1))
            p_x_mat *= 1 - torch.diag_embed(torch.ones((N, self.k_max)).to(self.device))
            p_x_mat_diag = torch.exp(logp_x_zt_two[:, t, :])
            prev_state = torch.argmax(alpha[:, t - 1, :], dim=-1)
            p_x_mat_diag[np.arange(N), prev_state] = torch.exp(logp_x_zt_cont[np.arange(N), prev_state])
            p_x_mat += torch.diag_embed(p_x_mat_diag)

            alpha[:, t, :] = torch.bmm(alpha[:, t - 1, :].unsqueeze(1), torch.mul(A.unsqueeze(0), p_x_mat+1e-25))[:, 0, :]

            state_persist = (torch.argmax(alpha[:, t - 1, :], dim=-1) == torch.argmax(alpha[:, t, :], dim=-1))
            z_count = torch.where(state_persist, (z_count + 1), torch.ones(N).to(self.device))
            logp_x_z_t = torch.stack([logp_x_zt_new[:, t, :], logp_x_zt_cont])[state_persist * 1, np.arange(N)]
            logp_x.append(logp_x_z_t)
            cs = cs + torch.log(torch.sum(alpha[:, t, :], -1))
            alpha[:, t, :] = alpha[:, t, :] / torch.sum(alpha[:, t, :], -1, keepdim=True)
            observed_likelihood = torch.where(lens - 1 == t,
                                              cs,
                                              torch.zeros(N, ).to(self.device))
            log_like += observed_likelihood
        if forward_only:
            return log_like

        ## Backward
        beta = torch.ones((N, T, self.k_max)).to(self.device)
        beta[:, -1, :] = beta[:, -1, :] / torch.sum(beta[:, -1, :], -1, keepdim=True)
        for t in range(T - 2, -1, -1):
            logp_x_z_t = logp_x[t + 1]
            p_x_z_t = torch.exp(logp_x_z_t) + 1e-25
            beta[:, t, :] = torch.matmul(A, (beta[:, t + 1, :] * p_x_z_t).T).T
            beta[:, t, :] = beta[:, t, :] / torch.sum(beta[:, t, :], -1, keepdim=True)
            beta[:, t, :] = torch.where(lens.unsqueeze(-1) - 1 > t,
                                        beta[:, t, :],
                                        torch.ones(N, self.k_max).to(self.device)/self.k_max)
        return (alpha * beta) / torch.sum((alpha * beta), axis=-1, keepdim=True), log_like

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import configparser
import os
import sys
import time
import torch

from utils import evaluate_dist, sample_posterior, load_data, \
    plot_state_predictions_test_samples, global_posterior_likelihood
from generator import HDPHMM
from bnp import HDP_FLOW

sys.path.append('..')
torch.autograd.set_detect_anomaly(True)
sns.set();
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device ....", device)

# Different simplified setups
use_nf = True


def main(args):
    np.random.seed(1234)
    if not os.path.exists("./plots/%s"%args.data):
        os.mkdir("./plots/%s"%args.data)
    config = configparser.ConfigParser()
    config.read('experiment_config.ini')

    if args.data not in config:
        raise ValueError("Configurations for experiments on %s don't exist"%args.data)
    lr_init = float(config[args.data]['lr'])
    mbatch_size = int(config[args.data]['mbatch_size'])
    k_max = int(config[args.data]['k_max'])
    n_features = int(config[args.data]['n_features'])
    n_epochs = int(config[args.data]['n_epochs'])
    n_mc_sample = int(config[args.data]['n_mc_sample'])
    periodic = str(config[args.data]['periodic'])=='True'
    decay = 0.1

    # Load data
    x_train, z_train, train_lens, x_valid, z_valid, valid_lens, x_test, z_test, test_lens = load_data(args.data, config,
                                                                                                      normalize=False)
    run_n = 'a%fg%fr%d%dl%fmade%d%d_%d' \
            % (args.alpha, args.gamma, args.rho[0], args.rho[1], args.lamda, args.made_blocks, args.made_size, args.run)

    # Train the model
    if args.train:
        if args.cont:
            bvi = torch.load("./ckpt/vi_%s_%s.pt" %(args.data, run_n), map_location=torch.device(device))
        else:
            gen_model = HDPHMM(alpha=args.alpha, gma=args.gamma, rho=[args.rho[0], args.rho[1]], lamda=[0, args.lamda],
                               n_features=n_features, MADE_hidden_size=args.made_size, n_MADE_blocks=args.made_blocks,
                               MADE_batch_norm=True, device=device, periodic=periodic)
            bvi = HDP_FLOW(prior_model=gen_model, data=x_train, k_max=k_max, device=device)
        ckpt_name = "vi_%s_%s" %(args.data, run_n)
        start_time = time.perf_counter()
        bvi.train(x_train, train_lens, n_epochs=n_epochs, n_mc_sample=n_mc_sample, lr_init=lr_init, mbatch_size=mbatch_size,
                  data=args.data, decay=decay, ckpt_name=ckpt_name)
        end_time = time.perf_counter()
        train_time = end_time - start_time
        print(">>>>>>> Training time: %d hours, %d minutes, %d seconds"%(train_time//3600,
                                                                         (train_time % 3600)//60,
                                                                         train_time % 60))

    # Evaluate performance
    bvi = torch.load("./ckpt/vi_%s_%s.pt" %(args.data, run_n), map_location=torch.device(device))
    q_z, q_pi, q_theta, q_beta, q_kappa = bvi._get_q_dist()
    state_probs = q_pi.sample((100,)).mean(0)
    init_probs = q_beta.sample((100,)).mean(0)
    kappa = q_kappa.sample((100,)).mean(0)
    print('Beta params: ', bvi.beta_param)
    print('Init prob', init_probs)
    print('Theta: ', np.sort(abs(bvi.theta_mu.cpu().detach().numpy().reshape(-1,)))[::-1][:10])
    print('Pi params: ', bvi.pi_param[0])
    print('kappa params: ', bvi.kappa_params)
    print('Kappas: ', kappa)
    state_probs = state_probs #* (1 - kappa).unsqueeze(-1) + torch.diag(kappa)
    ordered_states = np.argsort(init_probs.cpu().numpy())[::-1]

    state_posterior_logp_train,  _ = global_posterior_likelihood(model=bvi, data_labels=z_train, config=config[args.data])
    state_posterior_logp_test, _ = global_posterior_likelihood(model=bvi, data_labels=z_test, config=config[args.data])
    print(state_posterior_logp_train, state_posterior_logp_test)

    # Plot state transition matrix
    plt.figure()
    ordered_matrix = state_probs.cpu().detach().numpy().copy()[ordered_states, :]
    ordered_matrix = ordered_matrix[:, ordered_states]
    sns.heatmap(ordered_matrix)#state_probs.cpu().detach().numpy())
    plt.savefig("./plots/%s/transitions_matrix.pdf" %(args.data))

    gt_state_count = {}
    for z_s in (np.unique(z_train)):
        gt_state_count[z_s] = np.count_nonzero(z_train==z_s)
    if str(config[args.data]['padded']) == 'True':
        del gt_state_count[max(gt_state_count.keys())] # If padded, the last state is just padding
    gt_state_count = dict(sorted(gt_state_count.items(), key=lambda x: x[1], reverse=True))  # sort states


    # Visualize some results
    colors_pre = list(sns.color_palette('Set3', k_max))
    colors = colors_pre.copy()

    for st_ind, st in enumerate(ordered_states):
        colors[st] = colors_pre[st_ind]


    posterior_valid_all, posterior_test_all, acc_train_all, acc_test_all, acc_valid_all = [], [], [], [], []
    logp_z, predicted_z_test, posterior_test = bvi.find_state(torch.Tensor(x_test).to(device),
                                                              lens=torch.Tensor(test_lens).to(device), n_mc_sample=40)
    _, predicted_z_train, _ = bvi.find_state(torch.Tensor(x_train).to(device),
                                             lens=torch.Tensor(train_lens).to(device), n_mc_sample=40)
    _, predicted_z_valid, posterior_valid = bvi.find_state(torch.Tensor(x_valid).to(device),
                                                           lens=torch.Tensor(valid_lens).to(device), n_mc_sample=40)
    acc_train, mapping = evaluate_dist(torch.Tensor(z_train), predicted_z_train.cpu().detach().float(), train_lens, bvi.k_max)
    acc_test, _ = evaluate_dist(torch.Tensor(z_test), predicted_z_test.cpu().detach().float(), test_lens,
                                bvi.k_max, mapping=mapping)
    acc_valid, _ = evaluate_dist(torch.Tensor(z_valid), predicted_z_valid.cpu().detach().float(), valid_lens,
                                 bvi.k_max, mapping=mapping)

    print("*"*20)
    print("Log probability of predictive posterior (Test): %.2f +- %.2f" % ((np.mean(posterior_test), np.std(posterior_test))))
    print("Log probability of predictive posterior (Valid): %.2f +- %.2f" % ((np.mean(posterior_valid), np.std(posterior_valid))))
    print("*"*20)
    print("State prediction Hamming dist. (Train): %.3f +- %.3f" % (np.mean(acc_train), np.std(acc_train)))
    print("State prediction Hamming dist. (Test): %.3f +- %.3f" % (np.mean(acc_test), np.std(acc_test)))
    print("State prediction Hamming dist. (Valid): %.3f +- %.3f" % (np.mean(acc_valid), np.std(acc_valid)))

    plt.figure(figsize=(8, 2))
    prior_sample, states, sample_std = sample_posterior(bvi, init_probs)
    sample_std = np.clip(sample_std, a_min=None, a_max=10)
    generated_sample = prior_sample#np.clip(prior_sample, -30, +30)
    for feat in range(generated_sample.shape[1]):
        plt.plot(generated_sample[:, feat])
        plt.fill_between(np.arange(len(generated_sample)), (generated_sample[:, feat] - sample_std[:, feat]),
                         (generated_sample[:, feat] + sample_std[:, feat]), color='b', alpha=.4)
    for t in range(0, generated_sample.shape[0]-1):
        # plt.axvspan(t-1, t, alpha=0.4, color=colors[state_color_map[states[t]]])
        plt.axvspan(t, t+1, alpha=0.4, color=colors[states[t]])
    plt.ylim(x_train.min()-2, x_train.max()+2)
    plt.title("Generated Time series sample from the posterior", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("./plots/%s/posterior_generated.pdf" % (args.data))

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].pie(np.sort(init_probs.cpu().detach().numpy())[::-1],
               colors=[colors[ii_state] for ii_state in ordered_states])
               # colors=[colors[state_color_map[ii_state]] for ii_state in ordered_states])
    axs[0].set_title("Predicted class distribution")
    axs[1].pie(gt_state_count.values(),
               colors=[colors[mapping[ii_state]] for (ii_state) in gt_state_count.keys()])
    axs[1].set_title("Actual class distribution")
    plt.savefig("./plots/%s/state_dist.pdf" %(args.data))

    with open('%s_color_codes.pkl'%args.data, 'wb') as f:
        pickle.dump([colors[mapping[ii_state]] for (ii_state) in gt_state_count.keys()], f)

    plot_state_predictions_test_samples(n_test_samples=min(10, len(x_train)), test_x=x_train, test_z=z_train,
                                        test_lens=train_lens, test_preds=predicted_z_train, state_mapper=mapping,
                                        colors=colors, save_path="./plots/%s/ts_sample_train_IND.pdf" % (args.data))
    plot_state_predictions_test_samples(n_test_samples=min(10, len(x_test)), test_x=x_test, test_z=z_test,
                                        test_lens=test_lens, test_preds=predicted_z_test, state_mapper=mapping,
                                        colors=colors, save_path="./plots/%s/ts_sample_test_IND.pdf" % (args.data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run BVI')
    parser.add_argument('--data', type=str, default='sim_hard')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--cont', action='store_true')
    parser.add_argument('--alpha', type=float, default=30)
    parser.add_argument('--gamma', type=float, default=5)
    parser.add_argument('--rho', nargs=2, type=float, default=[5, 2])
    parser.add_argument('--lamda', type=float, default=1)
    parser.add_argument('--made_size', type=int, default=8)
    parser.add_argument('--made_blocks', type=int, default=1)
    parser.add_argument('--run', type=int, default=1)
    args = parser.parse_args()
    print("============= Experiment (%d) with Alpha %f, Gamma %f Rho %f,%f Lamda %f MADE %d %d ============="
          %(args.run, args.alpha, args.gamma, args.rho[0], args.rho[1], args.lamda, args.made_blocks, args.made_size))
    main(args)

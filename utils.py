import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import confusion_matrix, hamming_loss
from torch.nn.utils.rnn import pad_sequence
from torch.distributions.dirichlet import Dirichlet
import seaborn as sns
import copy


def evaluate_dist_beta(z_true, z_pred, beta, return_mapper=False):
    """
    Get accuracy after assigning states using the beta vector
    """
    if not isinstance(z_true, np.ndarray):
        z_true = np.array(z_true)
    if not isinstance(z_pred, np.ndarray):
        z_pred = np.array(z_pred)
    assert z_true.shape == z_pred.shape
    
    gt_states, counts = np.unique(z_true, return_counts=True)

    gt_states = gt_states[np.argsort(counts)[::-1]] # sorted with most common state first
    pred_sorted = np.argsort(beta)[::-1] # Predicted states, sorted with highest density first
    
    z_true_copy = copy.deepcopy(z_true)
    masks = []
    mapper = {}
    for state in gt_states:
        masks.append(z_true_copy==state)
    
    for i in range(len(gt_states)):
        z_true_copy[masks[i]] = pred_sorted[i]
        mapper[gt_states[i]] = pred_sorted[i]
    
    acc = hamming_loss(z_true_copy.reshape(-1,), z_pred.reshape(-1,))

    if return_mapper:
        return acc, mapper
    else:
        return acc


def evaluate_dist(z_true, z_pred, z_lens, k_max, mapping=None):
    """
    z_true: iterable of ground truth values for each dataset. eg [train_true, val_true, test_true]. Each dataset is some number of samples x T
    z_pred: same as above, but predictions from model
    k_max: upper limit to the number of states in the finite approximation
    return_mapped: true iff you want the mapped predictions in addition to accuracy
    """
    if not isinstance(z_true, np.ndarray):
        z_true = np.array(z_true)
    if not isinstance(z_pred, np.ndarray):
        z_pred = np.array(z_pred)
    
    
    true_vec = np.concatenate([zz[:z_lens[z_i]] for z_i, zz in enumerate(z_true)])
    pred_vec = np.concatenate([zz[:z_lens[z_i]] for z_i, zz in enumerate(z_pred)])
    cm = confusion_matrix(true_vec, pred_vec, labels=np.arange(k_max))  # the ij'th element is the number of class i predicted as class
    row_ind, col_ind = linear_assignment(cm, maximize=True)
    if mapping is None:
        mapping = {}
        for true_labels in np.unique(true_vec):
            mapping[int(true_labels)] = col_ind[int(true_labels)]

    z_true_mapped = np.copy(true_vec)
    for (gt_z, pred_z) in mapping.items():
         z_true_mapped[z_true_mapped == gt_z] = pred_z
    print(1 - (z_true_mapped == pred_vec).sum() / len(z_true_mapped))
    z_true_mapped = np.copy(z_true)
    for (gt_z, pred_z) in mapping.items():
        z_true_mapped[z_true == gt_z] = pred_z
    hamming_dist = 1 - np.array([(z_true_mapped[ii][:z_lens[ii]]==z_pred[ii][:z_lens[ii]]).sum()/z_lens[ii] for ii in range(len(z_true))])
    return hamming_dist, mapping

def evaluate_dist_other(z_true, z_pred, z_true_train, z_pred_train, k_max, mapping=None):
    if not isinstance(z_true, np.ndarray):
        z_true = np.array(z_true)
    if not isinstance(z_pred, np.ndarray):
        z_pred = np.array(z_pred)
    
    true_vec = z_true
    pred_vec = z_pred
    cm = confusion_matrix(z_true_train, z_pred_train, labels=np.arange(k_max))  # the ij'th element is the number of class i predicted as class
    row_ind, col_ind = linear_assignment(cm, maximize=True)
    if mapping is None:
        mapping = {}
        for true_labels in np.unique(true_vec):
            mapping[int(true_labels)] = col_ind[int(true_labels)]

    z_true_mapped = np.copy(true_vec)
    for (gt_z, pred_z) in mapping.items():
         z_true_mapped[z_true_mapped == gt_z] = pred_z
    print(1 - (z_true_mapped == pred_vec).sum() / len(z_true_mapped))
    return 



def evaluate_dist_w_uncertainty(z_true, z_pred, z_lens, k_max,
                                uncertainty_measures, thresholds, rm_indices, mapping=None):
    if not isinstance(z_true, np.ndarray):
        z_true = np.array(z_true)
    if not isinstance(z_pred, np.ndarray):
        z_pred = np.array(z_pred)

    true_vec, pred_vec, uncertainty_vec = [], [], []
    
    for index in range(len(z_true)):
        if rm_indices is not None and index in rm_indices:
            continue
        
        zt_part = z_true[index, :z_lens[index]]
        zp_part = z_pred[index, :z_lens[index]]

        # Apply uncertainty filtering
        mask_gamma = uncertainty_measures['gamma_var'][index, :z_lens[index]] < thresholds['gamma_var']
        # Additional filtering (if needed)
        # mask_entropy = uncertainty_measures['state_entropy'][index, :z_lens[index]] < thresholds['state_entropy']
        mask_credible_interval = uncertainty_measures['credible_interval_width'][index, :z_lens[index]] < thresholds['credible_interval_width']

        valid_mask = mask_gamma & mask_credible_interval# Can be combined with other masks using &
        zt_part = zt_part[valid_mask]
        zp_part = zp_part[valid_mask]
        uncertainty_part = uncertainty_measures['gamma_var'][index, :z_lens[index]][valid_mask]
        uncertainty_part += uncertainty_measures['credible_interval_width'][index, :z_lens[index]][valid_mask]
        true_vec.extend(zt_part)
        pred_vec.extend(zp_part)
        uncertainty_vec.extend(uncertainty_part)
    true_vec = np.array(true_vec)
    pred_vec = np.array(pred_vec)
    uncertainty_vec = np.array(uncertainty_vec)
    # Handle case where all filtered data is removed
    if len(true_vec) == 0:
        return np.array([]), mapping

    cm = confusion_matrix(true_vec, pred_vec, labels=np.arange(k_max))  # the ij'th element is the number of class i predicted as class
    row_ind, col_ind = linear_assignment(cm, maximize=True)
    if mapping is None:
        mapping = {int(true_labels): col_ind[int(true_labels)] for true_labels in np.unique(true_vec)}

    z_true_mapped = np.copy(true_vec)
    for (gt_z, pred_z) in mapping.items():
         z_true_mapped[z_true_mapped == gt_z] = pred_z
    print(1 - (z_true_mapped == pred_vec).sum() / len(z_true_mapped))
    z_true_mapped = np.copy(z_true)
    for (gt_z, pred_z) in mapping.items():
        z_true_mapped[z_true == gt_z] = pred_z
    hamming_dist = 1 - np.array([(z_true_mapped[ii][:z_lens[ii]]==z_pred[ii][:z_lens[ii]]).sum()/z_lens[ii] for ii in range(len(z_true))])
    return hamming_dist, mapping

def sample_posterior(model, init_probs):
    _, _, q_theta, q_beta, _ = model._get_q_dist()
    if isinstance(init_probs, np.ndarray):  # Check if already a NumPy array
        ordered_states = np.argsort(init_probs)[::-1]
    else:
        ordered_states = np.argsort(init_probs.detach().cpu().numpy())[::-1]

    thetas = model.theta_mu
    sample, states, sample_std = [], [], []
    for state in ordered_states:
        if isinstance(init_probs, np.ndarray):  # Check if already a NumPy array
            ratio = init_probs.copy()[state]
        else:  # Assume it's a tensor
            ratio = init_probs.detach().numpy().copy()[state]
        delta = 1
        model.prior_model.fill_flow_params(theta_vector=thetas[state])
        for t in range(int(ratio*model.T)):
            x_t = model.prior_model.flows.sample(num_samples=40, context=torch.Tensor([(delta)]).unsqueeze(-1))[0]
            sample.append(torch.mean(x_t, axis=0).cpu().detach().numpy())
            sample_std.append(torch.std(x_t, axis=0).cpu().detach().numpy())
            delta += 1
            states.append(state)
    return np.array(sample), np.array(states), np.array(sample_std)


def global_posterior_likelihood(model, data_labels, config, flag_config=True):
    sorted_params = np.sort(model.beta_param.detach().numpy())[::-1]
    q_beta = Dirichlet((torch.nn.Softmax(-1)(torch.Tensor(sorted_params.copy()))*200+1e-10))
    gt_state_count = {}
    for z_s in (np.unique(data_labels)):
        gt_state_count[z_s] = np.count_nonzero(data_labels == z_s)
    if flag_config:
        if str(config['padded']) == 'True':
            del gt_state_count[max(gt_state_count.keys())]  # If padded, the last state is just padding
    gt_state_count = dict(sorted(gt_state_count.items(), key=lambda x: x[1], reverse=True))  # sort states
    gt_probs = torch.zeros(len(model.beta_param))+1e-10
    gt_probs[:len(gt_state_count)] = torch.Tensor(list(gt_state_count.values()))
    state_logp = q_beta.log_prob(gt_probs/sum(gt_probs))

    ref_b = torch.zeros_like(model.beta_param)
    ref_b[:len(gt_state_count)] = torch.Tensor(list(gt_state_count.values()))
    q_beta_ref = Dirichlet(ref_b/sum(ref_b) + 1e-10)
    log_p_ref = q_beta_ref.log_prob(gt_probs/sum(gt_probs))
    return state_logp, log_p_ref


def load_data(data_type, config, normalize=False, pad_ragged=True, path_to_data='./data/',
              feat_size=None, wo_sleep=False, flag_external=False, flag_mean_norm=False):
    if data_type == 'sim_easy':
        n_train = int(config[data_type]['n_train'])
        n_valid = int(config[data_type]['n_valid'])
        n_test = int(config[data_type]['n_test'])
        x_all = np.load(path_to_data+'sim_easy_x.npy')
        z_all = np.load(path_to_data+'sim_easy_z.npy')
        T = x_all.shape[1]
        x_test = x_all[n_train + n_valid:n_train + n_valid + n_test]
        z_test = z_all[n_train + n_valid:n_train + n_valid + n_test]
        x_valid = x_all[n_train:n_train + n_valid]
        z_valid = z_all[n_train:n_train + n_valid]
        x_train = x_all[:n_train]
        z_train = z_all[:n_train]
        train_lens, valid_lens, test_lens = [T] * len(x_train), [T] * len(x_valid), [T] * len(x_test)

    elif data_type == 'sim_hard':
        n_train = int(config[data_type]['n_train'])
        n_valid = int(config[data_type]['n_valid'])
        n_test = int(config[data_type]['n_test'])
        x_all = np.load(path_to_data+'sim_hard_x.npy')
        z_all = np.load(path_to_data+'sim_hard_z.npy')
        T = x_all.shape[1]
        x_test = x_all[n_train + n_valid:n_train + n_valid + n_test]
        z_test = z_all[n_train + n_valid:n_train + n_valid + n_test]
        x_valid = x_all[n_train:n_train + n_valid]
        z_valid = z_all[n_train:n_train + n_valid]
        x_train = x_all[:n_train]
        z_train = z_all[:n_train]
        train_lens, valid_lens, test_lens = [T] * len(x_train), [T] * len(x_valid), [T] * len(x_test)

    elif data_type == 'sim_semi_markov':
        n_train = int(config[data_type]['n_train'])
        n_valid = int(config[data_type]['n_valid'])
        n_test = int(config[data_type]['n_test'])
        x_all = np.load(path_to_data + 'sim_semi_markov_x.npy')
        z_all = np.load(path_to_data + 'sim_semi_markov_z.npy')
        T = x_all.shape[1]
        x_test = x_all[n_train + n_valid:n_train + n_valid + n_test]
        z_test = z_all[n_train + n_valid:n_train + n_valid + n_test]
        x_valid = x_all[n_train:n_train + n_valid]
        z_valid = z_all[n_train:n_train + n_valid]
        x_train = x_all[:n_train]
        z_train = z_all[:n_train]
        train_lens, valid_lens, test_lens = [T] * len(x_train), [T] * len(x_valid), [T] * len(x_test)

    elif data_type in ['har', 'har_large']:
        x_all = np.load("./data/train_data.npy", allow_pickle=True)
        print('Training samples: ', len(x_all))
        z_all = np.load("./data/train_labels.npy", allow_pickle=True)
        x_test_all = np.load("./data/test_data.npy", allow_pickle=True)
        z_test_all = np.load("./data/test_labels.npy", allow_pickle=True)
        n_train = int(0.8 * len(x_all))
        if data_type == 'har':
            ds_factor = int(config[data_type]['ds_factor'])
        elif data_type == 'har_large': 
            ds_factor = 25
        n_features = x_all[0].shape[1]
        n_classes = len(np.unique(np.concatenate(z_all)))
        test_lens = [len(xx)//ds_factor for xx in x_test_all]
        z_test, x_test = [], []
        for zz, xx in zip(z_test_all, x_test_all):
            if len(zz) % ds_factor == 0:
                z_test.append(torch.Tensor([np.argmax(np.bincount(labels.astype(int))) for labels in (zz.reshape(-1, ds_factor))]))
                x_test.append(torch.Tensor(xx.reshape(-1, ds_factor, n_features).mean(axis=1)))
            else:
                z_test.append(torch.Tensor([np.argmax(np.bincount(labels.astype(int))) for labels in (zz[:-(len(zz) % ds_factor)].reshape(-1, ds_factor))]))
                x_test.append(torch.Tensor(xx[:-(len(xx) % ds_factor)].reshape(-1, ds_factor, n_features).mean(axis=1)))
        if pad_ragged:
            z_test = pad_sequence(z_test, batch_first=True, padding_value=n_classes)
            x_test = pad_sequence(x_test, batch_first=True)
        else:
            z_test = np.array([np.array(zz) for zz in z_test], dtype=object)
            x_test = np.array([np.array(xx) for xx in x_test], dtype=object)


        valid_lens = [len(xx)//ds_factor for xx in x_all[n_train:]]
        z_valid, x_valid = [], []
        for zz, xx in zip(z_all[n_train:], x_all[n_train:]):
            if len(zz) % ds_factor == 0:
                z_valid.append(torch.Tensor([np.argmax(np.bincount(labels.astype(int))) for labels in (zz.reshape(-1, ds_factor))]))
                x_valid.append(torch.Tensor(xx.reshape(-1, ds_factor, n_features).mean(axis=1)))
            else:
                z_valid.append(torch.Tensor([np.argmax(np.bincount(labels.astype(int))) for labels in (zz[:-(len(zz) % ds_factor)].reshape(-1, ds_factor))]))
                x_valid.append(torch.Tensor(xx[:-(len(xx) % ds_factor)].reshape(-1, ds_factor, n_features).mean(axis=1)))
        if pad_ragged:
            z_valid = pad_sequence(z_valid, batch_first=True, padding_value=n_classes)
            x_valid = pad_sequence(x_valid, batch_first=True)
        else:
            z_valid = np.array([np.array(zz) for zz in z_valid], dtype=object)
            x_valid = np.array([np.array(xx) for xx in x_valid], dtype=object)

        train_lens = [len(xx)//ds_factor for xx in x_all[:n_train]]
        z_train, x_train = [], []
        for zz, xx in zip(z_all[:n_train], x_all[:n_train]):
            if len(zz) % ds_factor == 0:
                z_train.append(torch.Tensor([np.argmax(np.bincount(labels.astype(int))) for labels in (zz.reshape(-1, ds_factor))]))
                x_train.append(torch.Tensor(xx.reshape(-1, ds_factor, n_features).mean(axis=1)))
            else:
                z_train.append(torch.Tensor([np.argmax(np.bincount(labels.astype(int))) for labels in (zz[:-(len(zz) % ds_factor)].reshape(-1, ds_factor))]))
                x_train.append(torch.Tensor(xx[:-(len(xx) % ds_factor)].reshape(-1, ds_factor, n_features).mean(axis=1)))
        if pad_ragged:
            z_train = pad_sequence(z_train, batch_first=True, padding_value=n_classes)
            x_train = pad_sequence(x_train, batch_first=True)
        else:
            z_train = np.array([np.array(zz) for zz in z_train], dtype=object)
            x_train = np.array([np.array(xx) for xx in x_train], dtype=object)

    elif data_type == 'har_70':
        subject_dfs = []
        ds_factor = int(config[data_type]['ds_factor'])
        feature_list = ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]
        for filename in os.listdir(path_to_data+'har_70'):
            if filename.endswith(".csv"):
                # Construct the full file path
                file_path = os.path.join(path_to_data+'har_70', filename)
                df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col=['timestamp'])
                df = df.ffill()
                if (len(df)%ds_factor)!= 0:
                    df = df[:-(len(df)%ds_factor)]
                if len(df) < ds_factor:
                    continue
                df_data = [x_bin.mean(0) for x_bin in df[feature_list].to_numpy().reshape(-1, ds_factor, len(feature_list))]
                df_label = [np.argmax(np.bincount(label_bin)) for label_bin in df["label"].to_numpy().reshape(-1, ds_factor)]
                df_final = pd.DataFrame(columns=feature_list + ["label"])
                df_final["label"] = df_label
                df_final[feature_list] = np.stack(df_data)
                subject_dfs.append(df_final)
        n_train, n_valid, n_test = 10, 3, 5
        n_classes = 8
        train_lens = [len(xx) for xx in subject_dfs[:n_train]]
        valid_lens = [len(xx) for xx in subject_dfs[n_train:n_train+n_valid]]
        test_lens = [len(xx) for xx in subject_dfs[-n_test:]]
        if pad_ragged:
            x_train = pad_sequence([torch.Tensor(xx[feature_list].to_numpy()) for xx in subject_dfs[:n_train]], batch_first=True)
            z_train = pad_sequence([torch.Tensor(xx['label'].to_numpy()) for xx in subject_dfs[:n_train]], batch_first=True,
                                padding_value=n_classes)
        
            x_valid = pad_sequence([torch.Tensor(xx[feature_list].to_numpy()) for xx in subject_dfs[n_train:n_train+n_valid]], batch_first=True)
            z_valid = pad_sequence([torch.Tensor(xx['label'].to_numpy()) for xx in subject_dfs[n_train:n_train+n_valid]],
                                batch_first=True, padding_value=n_classes)
        
            x_test = pad_sequence([torch.Tensor(xx[feature_list].to_numpy()) for xx in subject_dfs[-n_test:]], batch_first=True)
            z_test = pad_sequence([torch.Tensor(xx['label'].to_numpy())  for xx in subject_dfs[-n_test:]], batch_first=True,
                                padding_value=n_classes)
        
        else: # return unpadded np arrays
            x_train = np.array([xx[feature_list].to_numpy() for xx in subject_dfs[:n_train]], dtype=object)
            z_train = np.array([xx['label'].to_numpy() for xx in subject_dfs[:n_train]], dtype=object)
        
            x_valid = np.array([xx[feature_list].to_numpy() for xx in subject_dfs[n_train:n_train+n_valid]], dtype=object)
            z_valid = np.array([xx['label'].to_numpy() for xx in subject_dfs[n_train:n_train+n_valid]], dtype=object)
        
            x_test = np.array([xx[feature_list].to_numpy() for xx in subject_dfs[-n_test:]], dtype=object)
            z_test = np.array([xx['label'].to_numpy()  for xx in subject_dfs[-n_test:]], dtype=object)

    elif data_type == 'cpap':
        x_train = np.load(path_to_data+'CPAP_train_data.npy', allow_pickle=True)
        z_train = np.load(path_to_data+'CPAP_train_labels.npy', allow_pickle=True)
        x_test = np.load(path_to_data+'CPAP_test_data.npy', allow_pickle=True)
        z_test = np.load(path_to_data+'CPAP_test_labels.npy', allow_pickle=True)
        
        n_train = int(0.8 * len(x_train))
        n_classes = len(np.unique(np.concatenate(z_train)))
        test_lens = [len(xx) for xx in x_test]
        valid_lens = [len(xx) for xx in x_train[n_train:]]
        train_lens = [len(xx) for xx in x_train[1:n_train]]  # removing sample 1 because of an issue in the data
        if pad_ragged:
            x_test = pad_sequence([torch.Tensor(xx) for xx in (x_test)], batch_first=True)
            z_test = pad_sequence([torch.Tensor(xx) for xx in (z_test)], batch_first=True, padding_value=n_classes)
            
            x_valid = pad_sequence([torch.Tensor(xx) for xx in (x_train[n_train:])], batch_first=True)
            z_valid = pad_sequence([torch.Tensor(xx) for xx in (z_train[n_train:])], batch_first=True, padding_value=n_classes)
            
            x_train = pad_sequence([torch.Tensor(xx) for xx in (x_train[1:n_train])], batch_first=True)
            z_train = pad_sequence([torch.Tensor(xx) for xx in (z_train[1:n_train])], batch_first=True, padding_value=n_classes)
        
        else: 
            x_test = np.array([xx for xx in (x_test)], dtype=object)
            z_test = np.array([xx for xx in (z_test)], dtype=object)
            
            x_valid = np.array([xx for xx in (x_train[n_train:])], dtype=object)
            z_valid = np.array([xx for xx in (z_train[n_train:])], dtype=object)
            
            x_train = np.array([xx for xx in (x_train[1:n_train])], dtype=object)
            z_train = np.array([xx for xx in (z_train[1:n_train])], dtype=object)
            
    elif data_type == 'bump' or data_type == 'crohns':
        #Batch * feature * T shape
        if feat_size==12 or feat_size==None:
            save_name = 'wstress'
        else:
            save_name = 'objonly' 
        x_both = np.load('../data/'+data_type+'_features_'+save_name+'.npy', allow_pickle=True) #full length of data=290 #bump_features_smoothed
        x_both = np.transpose(x_both, (0, 2, 1))
        if wo_sleep:
            feat_list = ['deep', 'hr_average', 'light', 'midpoint_at_delta', 'onset_latency', 'rem', 'rmssd', 
                     'score', 'temperature_delta', 'awake']
            feat_ids = [1, 3, 6, 8]
            x_both = x_both[:,:,feat_ids]
        #feat_ids = [1,4,5,6,7,8,10,11,13,14]
        #x_both = x_both[:,:,feat_ids]
        z_both = np.load('../data/'+data_type+'_labels_'+save_name+'.npy', allow_pickle=True) 
        len_both = np.load('../data/'+data_type+'_data_len_'+save_name+'.npy', allow_pickle=True)
        if flag_external:
            imput_values = [4706.68358714044, 71.55861687786465, 15392.61889066767, 10205.761047002534, 651.895105539681, 5653.638157894735, 30.841147786946745, 77.9042818829187, -0.06723937318194617, 4084.0870209220516]
            for i in range(10):
                for j in range(len(len_both)):
                    indices = [False]*z_both.shape[1]
                    indices[:len_both[j]] = (z_both[j,:len_both[j]] == 5)
                    if sum(indices) >0: 
                        #print(i, j, np.mean(x_both[j, indices, i]))
                        x_both[j, indices, i] = imput_values[i]
                        
        flag_merge_labels = True
        y_both_new = np.zeros_like(z_both)
        if flag_merge_labels:
            y_both_new[(0<z_both) & (z_both<=3)] = 1
            y_both_new[z_both>3] = z_both[z_both>3] - 2
        z_both = y_both_new
        
        n_train = int(0.8 * len(x_both))
        x_train, z_train, train_lens = x_both[:n_train], z_both[:n_train], len_both[:n_train]
        x_test, z_test, test_lens = x_both[n_train:], z_both[n_train:], len_both[n_train:]
        
        n_train = int(0.8 * len(x_train))
        n_classes = len(np.unique(np.concatenate(z_train)))
        x_valid, z_valid, valid_lens = x_train[n_train:], z_train[n_train:], train_lens[n_train:]
        x_train, z_train, train_lens = x_train[:n_train], z_train[:n_train], train_lens[:n_train]
        
        if normalize:
            N, T, feat_dim = x_train.shape
            if flag_mean_norm:
                if flag_external:
                    np_mean = [3039.62, 45.71, 9650.82, 6680.52, 403.56, 3706.49, 20.95, 49.24, -0.04, 2642.06]
                    np_std = [2892.25, 35.62, 7797.93, 5778.6, 437.18, 3324.08, 20.0, 38.36, 0.12, 2536.76]
                else:
                    np_mean = np.mean(x_train.reshape(-1,feat_dim), axis=0).reshape(1,1,-1)
                    np_std = np.std(x_train.reshape(-1,feat_dim), axis=0).reshape(1,1,-1)
                    print('means: ', list(np_mean.reshape(-1).round(2)))
                    print('std: ', list(np_std.reshape(-1).round(2))) 
                x_train = (x_train - np_mean) / np_std
                x_test = (x_test - np_mean) / np_std
                x_valid = (x_valid - np_mean) / np_std
            else:
                if flag_external:
                    mins = [ 0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,   -1.88,  0.  ]
                    maxes = [9.1800e+03, 8.9654e+01, 2.1384e+04, 1.8090e+04, 1.8900e+03, 1.1016e+04,
     6.8000e+01, 9.1000e+01, 2.0000e-01, 1.0116e+04] #Bump data mins and maxes 
                else:
                    mins = np.min(x_train.reshape(-1,feat_dim), axis=0)
                    maxes = np.max(x_train.reshape(-1,feat_dim), axis=0)
                for i in range(feat_dim):
                    if (maxes[i]-mins[i])>10:
                        x_train[:,:,i] = 10 * (x_train[:,:,i] - mins[i]) / (maxes[i] - mins[i])
                        x_test[:,:,i] = 10 * (x_test[:,:,i] - mins[i]) / (maxes[i] - mins[i])
                        x_valid[:,:,i] = 10 * (x_valid[:,:,i] - mins[i]) / (maxes[i] - mins[i])
        
    if normalize and (data_type not in ['bump', 'crohns']):
        feature_mean = x_train.reshape((-1, x_train.shape[-1])).mean(0)
        feature_std = x_train.reshape((-1, x_train.shape[-1])).std(0)
        x_train = (x_train - feature_mean) / feature_std
        x_test = (x_test - feature_mean) / feature_std
        x_valid = (x_valid - feature_mean) / feature_std

    train_lens = [int(l) for l in train_lens]
    valid_lens = [int(l) for l in valid_lens]
    test_lens = [int(l) for l in test_lens]
    print('length train val and test data: ', len(x_train), len(x_valid), len(x_test))
    return x_train, z_train, train_lens, x_valid, z_valid, valid_lens, x_test, z_test, test_lens


def plot_state_predictions_test_samples(n_test_samples, test_x, test_z, test_lens, test_preds, state_mapper, colors, save_path, feature_id=None, flag_color=False):
    for test_ind in range(n_test_samples):
        if flag_color:
            fig, axs = plt.subplots(3, 1, figsize=(10, 9))
        else:
            fig, axs = plt.subplots(2, 1, figsize=(10, 4))
        test_sample = test_x[test_ind][:test_lens[test_ind]]
        test_z_gt = test_z[test_ind][:test_lens[test_ind]]
        
        if torch.is_tensor(test_sample): # Convert to np if needed to plot
            test_sample = test_sample.cpu().detach().numpy()
            test_z_gt = test_z_gt.cpu().detach().numpy()
        test_pred = test_preds[test_ind][:test_lens[test_ind]]
        assert test_pred.shape == test_z_gt.shape

        for i in range(test_sample.shape[-1]):
            if feature_id == None:
                axs[0].plot(test_sample[:, i], label=f"Feature {i}")
                axs[1].plot(test_sample[:, i], label=f"Feature {i}")
            else:
                axs[0].plot(test_sample[:, i], label=f"Feature {feature_id[i]}")
                axs[1].plot(test_sample[:, i], label=f"Feature {feature_id[i]}")
        for t in range(test_sample.shape[0] - 1):
            axs[0].axvspan(t, t + 1, alpha=0.1, color=colors[state_mapper[test_z_gt[t]]])
            axs[1].axvspan(t, t + 1, alpha=0.1,
                           color=colors[int(test_pred[t])])
        axs[0].set_title("Ground truth underlying states")
        axs[1].set_title("Estimated underlying states")
        if flag_color:
            for i in range(len(state_mapper)):
                axs[2].plot([], [], color=colors[state_mapper[i]], label=i, linewidth=10)  # Dummy line for legend

            axs[2].legend(loc="center", ncol=len(colors))
            axs[2].axis('off')  # Hide axes for the color legend plot

        plt.tight_layout() 
        plt.legend()
        plt.savefig(save_path.replace('IND', str(test_ind)))
        plt.close()

def plot_new_state_predictions_samples(test_x, test_pred, test_y, test_len, feature_colors, feat_list):
            #test_pred = test_preds[test_ind][:test_lens[test_ind]]
    fig, ax = plt.subplots(figsize=(15, 6))

    for feat in range(len(feat_list)):
        ax.plot(test_x[:, feat], color=feature_colors[feat], label=f'Feature {feat_list[feat]}')

    for t in range(0, generated_sample.shape[0] - 1):
        ax.axvspan(t, t + 1, alpha=0.2, color=colors[int(test_pred[t])])

    ax.set_title("Features and their predicted states", fontsize=16, fontweight='bold')
    plt.tight_layout()

    legend_fig, legend_ax = plt.subplots(figsize=(10, 1))
    legend_ax.axis('off')  # Turn off the axes for the legend figure
    num_features = len(feat_list)
    half_features = num_features // 2
    first_row_items = [
        plt.Rectangle((0, 0), 1, 1, color=feature_colors[feat], label=f'Feature {feat_list[feat]}') 
        for feat in range(half_features)
    ]
    second_row_items = [
        plt.Rectangle((0, 0), 1, 1, color=feature_colors[feat], label=f'Feature {feat_list[feat]}') 
        for feat in range(half_features, num_features)
    ]
    legend_ax.legend(
        handles=first_row_items + second_row_items, 
        loc='center', 
        ncol=half_features, 
        frameon=False,
        prop={'size': 14}  # Larger font size
    )
    plt.show()       
    
        
def show_state_predictions(n_test_samples, test_x, test_z, test_lens, test_preds, state_mapper, colors, save_path, feature_id=None, flag_color=False):
    for test_ind in range(n_test_samples):
        if flag_color:
            fig, axs = plt.subplots(3, 1, figsize=(10, 9))
        else:
            fig, axs = plt.subplots(2, 1, figsize=(10, 4))
        test_sample = test_x[test_ind][:test_lens[test_ind]]
        test_z_gt = test_z[test_ind][:test_lens[test_ind]]
        
        if torch.is_tensor(test_sample): # Convert to np if needed to plot
            test_sample = test_sample.cpu().detach().numpy()
            test_z_gt = test_z_gt.cpu().detach().numpy()
        test_pred = test_preds[test_ind][:test_lens[test_ind]]
        assert test_pred.shape == test_z_gt.shape

        for i in range(test_sample.shape[-1]):
            if feature_id == None:
                axs[0].plot(test_sample[:, i], label=f"Feature {i}")
                axs[1].plot(test_sample[:, i], label=f"Feature {i}")
            else:
                axs[0].plot(test_sample[:, i], label=f"Feature {feature_id[i]}")
                axs[1].plot(test_sample[:, i], label=f"Feature {feature_id[i]}")
        for t in range(test_sample.shape[0] - 1):
            axs[0].axvspan(t, t + 1, alpha=0.1, color=colors[state_mapper[test_z_gt[t]]])
            axs[1].axvspan(t, t + 1, alpha=0.1,
                           color=colors[int(test_pred[t])])
        axs[0].set_title("Ground truth underlying states")
        axs[1].set_title("Estimated underlying states")
        if flag_color:
            for i in range(len(state_mapper)):
                axs[2].plot([], [], color=colors[state_mapper[i]], label=i, linewidth=10)  # Dummy line for legend

            axs[2].legend(loc="center", ncol=len(colors))
            axs[2].axis('off')  # Hide axes for the color legend plot

        plt.tight_layout() 
        plt.legend()
        plt.show()
        
def line_plot(values, filename, title='Line Plot', xlabel='Index', ylabel='Value'):
    """
    Plot an array of values and save the plot to a given filename.
    
    Parameters:
    - values: Array of values to be plotted.
    - filename: String indicating the name of the file to save the plot.
    """
    plt.figure()
    plt.plot(values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    # Save the plot to the specified filename
    plt.savefig(filename)
    plt.close()


if __name__ == '__main__':
    import seaborn as sns
    x = np.array([1.1, 2.2, 1.1, 3.3, 1.1, 4.4, 1.1, 5.5, 1.1, 2.2, 1.1, 3.3, 1.1, 4.4, 1.1, 5.5, 3, 2, 1, 0, 1]).reshape(1, -1, 1)
    z_true = np.array([[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0]])
    mapper = {0: 5, 1: 6, 2: 17, 3: 12}
    L = 20
    colors = sns.color_palette('Set3', L+1)
    z_pred = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    print(x.shape, z_true.shape, z_pred.shape)
    plot_state_predictions_test_samples(n_test_samples=1, 
                                        test_x=x, 
                                        test_z=z_true, 
                                        test_lens=[21], 
                                        test_preds=z_pred, 
                                        state_mapper=mapper, 
                                        colors=colors, 
                                        save_path='./TEST.pdf')
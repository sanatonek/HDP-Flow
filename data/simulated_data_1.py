import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def generate_samples(num_samples, sample_len):
    """
    params: 
    num_samples (int) is the number of samples to generate
    sample_len (int) is the length of each sample to generate

    output:
    samples, a np array of shape (num_samples, sample_len, num_features)
    true_states, a np array of shape (num_samples, sample_len)
    """
    samples = []
    true_states = []

    transition_probs = np.array([[0.8, 0.1, 0.05, 0.05],
                                [0.1, 0.8, 0.1, 0.],
                                [0.05, 0.1, 0.8, 0.05],
                                [0.05, 0.05, 0.0, 0.9]]) # Manually created transition matrix. the ij'th element is probability of changing from state i to state j


    state_means = np.array([[0, 1, 2], # mean of features for state 0
                            [5, 6, 1], # mean of features for state 1
                            [5., 5., 5],
                            [9, 12, 11.]]) # of shape (num_states, num_features). The ij'th element gives the mean of feature j when in state i. 

    var = 0.3
    for _ in range(num_samples):
        sample = []
        states = []
        # With uniform prob, select the first state
        curr_state = np.argmax(np.random.multinomial(n=1, pvals=np.ones(len(transition_probs))/len(transition_probs))) # Uniform probabilities 
        states.append(curr_state)
        x_t = np.random.multivariate_normal(mean=state_means[curr_state], cov=var*np.eye(state_means.shape[1])) # Returns sample of isotropic gaussian around the state mean vector
        sample.append(x_t)

        for t in range(1, sample_len): # Start at 1 because the first state was just selected
            curr_state = np.argmax(np.random.multinomial(n=1, pvals=transition_probs[curr_state])) # Select new state
            x_t = np.random.multivariate_normal(mean=state_means[curr_state], cov=var*np.eye(state_means.shape[1]))
            sample.append(x_t)
            states.append(curr_state)

        
        samples.append(np.array(sample))
        true_states.append(np.array(states))
    

    return np.stack(samples), np.stack(true_states)


if __name__ == '__main__':
    T = 80
    samples, states = generate_samples(650, T)
    x = samples[0]
    z = states[0]

    # Save data to .npy files
    with open('./sim_easy_x.npy', 'wb') as f:
        np.save(f, samples)
    with open('./sim_easy_z.npy', 'wb') as f:
        np.save(f, states)

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
    print('z_colors: ', z_colors)
    plt.title("Time series sample with underlying states")
    plt.savefig("./plots/simluated_ts_sample.pdf")




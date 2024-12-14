from sklearn.kernel_approximation import RBFSampler
import numpy as np

rbf_feature = RBFSampler(gamma=1, random_state=12345)


def extract_features(state, num_actions):
    """ 
    This function computes the RFF features for a BERT embedding state for all the discrete actions

    :param state: BERT embedding of the state (shape (embedding_dim,))
    :param num_actions: number of discrete actions to compute the RFF features for
    :return: phi(s,a) for all the actions (shape d x |num_actions|)
    """
    s = state.reshape(1, -1)
    s = np.repeat(s, num_actions, 0)  # Repeat state for each action
    a = np.arange(0, num_actions).reshape(-1, 1)  # Action indices
    sa = np.concatenate([s, a], axis=-1)  # Concatenate state-action pairs
    feats = rbf_feature.fit_transform(sa)  # Get RFF features
    feats = feats.T  # Transpose to match d x |num_actions|
    return feats


def compute_softmax(logits, axis):
    """ computes the softmax of the logits """
    logits -= np.max(logits, axis=axis, keepdims=True)
    exp_logits = np.exp(logits)
    softmax = exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)
    return softmax


def compute_action_distribution(theta, phis):
    """ compute probability distribution over actions """
    logits = np.dot(phis.T, theta)  # Shape: (num_actions, 1)
    return compute_softmax(logits, axis=0).T


def compute_log_softmax_grad(theta, phis, action_idx):
    """ computes the log softmax gradient for the action with index action_idx """
    action_dist = compute_action_distribution(theta, phis)
    expected_phi = np.dot(action_dist, phis.T)  # Shape: (d,)
    grad = phis[:, action_idx].reshape(-1, 1) - expected_phi.T  # Shape: (d, 1)
    return grad


def compute_fisher_matrix(grads, lamb=1e-3):
    """ computes the fisher information matrix using the sampled trajectories gradients """
    d = grads[0][0].shape[0]
    fisher_matrix = np.zeros((d, d))  # Initialize Fisher matrix

    for trajectory in grads:
        trajectory_fisher = np.zeros((d, d))
        for grad in trajectory:
            trajectory_fisher += np.dot(grad, grad.T)
        trajectory_fisher /= len(trajectory)  # Average over time steps
        fisher_matrix += trajectory_fisher

    fisher_matrix /= len(grads)  # Average over trajectories
    fisher_matrix += lamb * np.eye(d)  # Add regularization
    return fisher_matrix


def compute_value_gradient(grads, rewards):
    """ computes the value function gradient with respect to the sampled gradients and rewards """
    N = len(grads)
    total_rewards = [sum(trajectory_rewards) for trajectory_rewards in rewards]
    b = np.mean(total_rewards)
    value_gradient = np.zeros(grads[0][0].shape)
    
    for i in range(N):
        trajectory_sum = np.zeros(grads[0][0].shape)
        trajectory_gradients = grads[i]
        trajectory_rewards = rewards[i]
        H = len(trajectory_gradients)
        
        for h in range(H):
            reward_sum = np.sum(trajectory_rewards[h:])
            trajectory_sum += trajectory_gradients[h] * (reward_sum - b)
        
        trajectory_sum /= H
        value_gradient += trajectory_sum
    
    value_gradient /= N
    return value_gradient


def compute_eta(delta, fisher, v_grad):
    """ computes the learning rate for gradient descent """
    epsilon = 1e-6
    fisher_inv = np.linalg.inv(fisher)
    quadratic = np.dot(v_grad.T, np.dot(fisher_inv, v_grad))
    eta = np.sqrt(delta / (quadratic + epsilon))
    return eta

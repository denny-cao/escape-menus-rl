from sklearn.kernel_approximation import RBFSampler
import numpy as np

rbf_feature = RBFSampler(gamma=1, random_state=12345)


def extract_features(state, num_actions):
    """ This function computes the RFF features for a state for all the discrete actions

    :param state: column vector of the state we want to compute phi(s,a) of (shape |S|x1)
    :param num_actions: number of discrete actions you want to compute the RFF features for
    :return: phi(s,a) for all the actions (shape 100x|num_actions|)
    """
    s = state.reshape(1, -1)
    s = np.repeat(s, num_actions, 0)
    a = np.arange(0, num_actions).reshape(-1, 1)
    sa = np.concatenate([s,a], -1)
    feats = rbf_feature.fit_transform(sa)
    feats = feats.T
    return feats


def compute_softmax(logits, axis):
    """ computes the softmax of the logits

    :param logits: the vector to compute the softmax over
    :param axis: the axis we are summing over
    :return: the softmax of the vector

    Hint: to make the softmax more stable, subtract the max from the vector before applying softmax
    """

    return np.exp(logits - np.max(logits, axis=axis, keepdims=True)) / np.sum(np.exp(logits - np.max(logits, axis=axis, keepdims=True)), axis=axis, keepdims=True)


def compute_action_distribution(theta, phis):
    """ compute probability distrubtion over actions

    :param theta: model parameter (shape d x 1)
    :param phis: RFF features of the state and actions (shape d x |A|)
    :return: softmax probability distribution over actions (shape 1 x |A|)
    """

    logits = theta.T @ phis

    return compute_softmax(logits, axis=1)


def compute_log_softmax_grad(theta, phis, action_idx):
    """ computes the log softmax gradient for the action with index action_idx

    :param theta: model parameter (shape d x 1)
    :param phis: RFF features of the state and actions (shape d x |A|)
    :param action_idx: The index of the action you want to compute the gradient of theta with respect to
    :return: log softmax gradient (shape d x 1)
    """

    probs = compute_action_distribution(theta, phis)
    grad = phis[:, action_idx] - np.sum(phis * probs, axis=1) # \phi(s,a) - E_{a' \sim \pi_{\theta}}[\phi(s,a')]

    return grad.reshape(-1, 1)



def compute_fisher_matrix(grads, lamb=1e-3):
    """ computes the fisher information matrix using the sampled trajectories gradients

    :param grads: list of list of gradients, where each sublist represents a trajectory (each gradient has shape d x 1)
    :param lamb: lambda value used for regularization 

    :return: fisher information matrix (shape d x d)
    
    

    Note: don't forget to take into account that trajectories might have different lengths
    """

    d = grads[0][0].shape[0]
    N = len(grads)

    fisher_hat = np.zeros((d, d))

    # Sum of the outer products of the gradients
    for i in range(N):
        H = len(grads[i])
        for h in range(H):
            fisher_hat += (grads[i][h] @ grads[i][h].T) / H

    # Normalization
    fisher_hat /= N

    # Regularization
    fisher_hat += lamb * np.eye(d)

    return fisher_hat


def compute_value_gradient(grads, rewards):
    """ computes the value function gradient with respect to the sampled gradients and rewards

    :param grads: list of list of gradients, where each sublist represents a trajectory
    :param rewards: list of list of rewards, where each sublist represents a trajectory
    :return: value function gradient with respect to theta (shape d x 1)
    """

    N = len(grads)

    b = 1/N * np.sum([np.sum(rewards[i]) for i in range(N)])

    v_grad_hat = np.zeros(grads[0][0].shape)

    for i in range(N):
        H = len(grads[i])
        for j in range(H):
            rewards_sum = np.sum(rewards[i][j:])
            v_grad_hat += (rewards_sum - b) * grads[i][j] / H

    return v_grad_hat/N


def compute_eta(delta, fisher, v_grad):
    """ computes the learning rate for gradient descent

    :param delta: trust region size
    :param fisher: fisher information matrix (shape d x d)
    :param v_grad: value function gradient with respect to theta (shape d x 1)
    :return: the maximum learning rate that respects the trust region size delta
    """

    return np.sqrt(delta/(v_grad.T @ np.linalg.inv(fisher) @ v_grad + delta))

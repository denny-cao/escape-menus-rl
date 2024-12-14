import numpy as np

def compute_softmax(logits):
    max_logit = np.max(logits)
    exp_logits = np.exp(logits - max_logit)
    return exp_logits / np.sum(exp_logits)

def compute_action_distribution(theta, phi, num_actions):
    """

    theta: (A*M,) flattened
    phi: (M,)
    returns pi(s): (A,)
    """
    A = num_actions
    M = phi.shape[0]
    Theta = theta.reshape(A, M)  # A x M
    z = Theta.dot(phi)  # (A,)
    pi = compute_softmax(z)
    return pi

def compute_log_softmax_grad(theta, phi, action_idx, num_actions):
    """
    grad = (e_a - pi(s)) * phi(s)^T
    Shape:
      (e_a - pi) is (A,)
      phi(s) is (M,)
    outer product is (A, M) then flatten to (A*M, 1). see paper for more details
    """
    pi = compute_action_distribution(theta, phi, num_actions)
    A = num_actions


    e_a = np.zeros(A)
    e_a[action_idx] = 1.0

    diff = e_a - pi  # (A,)
    grad_matrix = np.outer(diff, phi)  # (A x M)
    grad = grad_matrix.flatten()[:, np.newaxis]  # (A*M, 1)
    return grad

def compute_fisher_matrix(grads, lamb=1e-3):
    """ computes the fisher information matrix using the sampled trajectories gradients """
    d = grads[0][0].shape[0]
    fisher_matrix = np.zeros((d, d))

    for trajectory in grads:
        trajectory_fisher = np.zeros((d, d))
        for grad in trajectory:
            trajectory_fisher += np.dot(grad, grad.T)
        trajectory_fisher /= len(trajectory)  # Average over steps
        fisher_matrix += trajectory_fisher

    fisher_matrix /= len(grads)  # Average over trajectories
    fisher_matrix += lamb * np.eye(d)
    return fisher_matrix

def compute_value_gradient(grads, rewards):
    """ computes the value function gradient with respect to the sampled gradients and rewards """
    N = len(grads)
    total_rewards = [sum(traj_rewards) for traj_rewards in rewards]
    b = np.mean(total_rewards)

    d = grads[0][0].shape[0]
    value_gradient = np.zeros((d, 1))

    for i in range(N):
        trajectory_sum = np.zeros((d, 1))
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

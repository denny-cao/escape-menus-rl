import numpy as np
import matplotlib.pyplot as plt
from sample_trajectory import TrajectorySampler
import utils
import os

def sample(theta, sampler, N, num_actions):
    total_rewards = []
    total_grads = []

    for _ in range(N):
        trajectory = sampler.sample_trajectory(max_steps=200)
        rewards = []
        grads = []
        
        for state, action, reward in trajectory:
            phi = state.numpy()  # (M,)
            grad = utils.compute_log_softmax_grad(theta, phi, action, num_actions)
            rewards.append(reward)
            grads.append(grad)
        
        total_rewards.append(rewards)
        total_grads.append(grads)

    return total_grads, total_rewards

def train(N, T, delta, sampler, num_actions=4, lamb=1e-3):
    # Initialize theta based on the embedding dimension
    init_state = sampler.reset()  # returns torch.Tensor
    M = init_state.shape[0]

    theta = np.random.randn(num_actions * M)  # Flattened parameter vector of size A*M
    episode_rewards = []

    for t in range(T):
        grads, rewards = sample(theta, sampler, N, num_actions)
        
        fisher_matrix = utils.compute_fisher_matrix(grads, lamb)
        v_grad = utils.compute_value_gradient(grads, rewards)
        eta = utils.compute_eta(delta, fisher_matrix, v_grad)

        fisher_inv = np.linalg.inv(fisher_matrix)
        theta += (eta * np.dot(fisher_inv, v_grad)).flatten()
        
        avg_reward = np.mean([sum(traj) for traj in rewards])
        episode_rewards.append(avg_reward)
        print(f"Iteration {t+1}: Avg Reward = {avg_reward:.2f}, ||v_grad|| = {np.linalg.norm(v_grad):.2f}")

    return theta, episode_rewards

if __name__ == '__main__':
    sampler = TrajectorySampler(json_folder="pr_50_br_3_dp_3")
    theta, episode_rewards = train(N=1000, T=100, delta=1e-5, sampler=sampler)

    plt.plot(episode_rewards)
    plt.title("Average Rewards per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Average Rewards")
    fig_dir = "fig"
    os.makedirs(fig_dir, exist_ok=True)
    
    # Save the plot inside the 'fig' directory
    plt.savefig(os.path.join(fig_dir, "average_rewards.png"), format="png")

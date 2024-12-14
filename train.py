import numpy as np
import utils
import matplotlib.pyplot as plt
from sample_trajectory import TrajectorySampler  # Use TrajectorySampler instead of gym


def sample(theta, sampler, N):
    total_rewards = []
    total_grads = []

    for _ in range(N):
        trajectory = sampler.sample_trajectory(max_steps=200)
        rewards = []
        grads = []
        
        for state, action, reward in trajectory:
            phis = utils.extract_features(state.numpy(), num_actions=4)  # Convert to NumPy
            grad = utils.compute_log_softmax_grad(theta, phis, action)  
            rewards.append(reward)
            grads.append(grad)
        
        total_rewards.append(rewards)
        total_grads.append(grads)

    return total_grads, total_rewards


def train(N, T, delta, sampler, lamb=1e-3):
    theta = np.random.rand(100, 1)  # Initialize policy parameters
    episode_rewards = []

    for t in range(T):
        # 1. Sample N trajectories
        grads, rewards = sample(theta, sampler, N)
        
        # 2. Compute the Fisher matrix
        fisher_matrix = utils.compute_fisher_matrix(grads, lamb)
        
        # 3. Compute the policy gradient
        v_grad = utils.compute_value_gradient(grads, rewards)
        
        # 4. Compute the step size
        eta = utils.compute_eta(delta, fisher_matrix, v_grad)
        
        # 5. Update the model parameters by taking an NPG step
        fisher_inv = np.linalg.inv(fisher_matrix)
        theta += eta * np.dot(fisher_inv, v_grad)
        
        avg_reward = np.mean([sum(trajectory_rewards) for trajectory_rewards in rewards])
        episode_rewards.append(avg_reward)
        print(f"Iteration {t+1}: Avg Reward = {avg_reward:.2f}, Loss = {np.linalg.norm(v_grad):.2f}")
    
    return theta, episode_rewards


if __name__ == '__main__':
    sampler = TrajectorySampler(json_folder="pr_25_br_3_dp_3")  # Use sampler
    theta, episode_rewards = train(N=100, T=20, delta=1e-2, sampler=sampler)
    plt.plot(episode_rewards)
    plt.title("Average Rewards per Timestep")
    plt.xlabel("Timestep")
    plt.ylabel("Average Rewards")
    plt.show()

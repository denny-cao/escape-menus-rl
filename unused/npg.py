# FAILED
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from sklearn.kernel_approximation import RBFSampler

class PolicyNetwork(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, features):
        x = torch.relu(self.fc1(features))
        logits = self.fc2(x)
        return logits

class NPGAgent:
    def __init__(self, feature_dim, action_dim, gamma=0.99, lamb=1e-3, delta=1e-2, device="cpu"):
        self.gamma = gamma
        self.lamb = lamb
        self.delta = delta
        self.device = device

        self.policy_network = PolicyNetwork(feature_dim, action_dim).to(device)

    def compute_action_distribution(self, theta, features):
        logits = features.T @ theta
        return torch.softmax(logits, dim=1)

    def compute_log_softmax_grad(self, theta, features, action_idx):
        probs = self.compute_action_distribution(theta, features)
        grad = features[:, action_idx] - torch.sum(features * probs, dim=1, keepdim=True)
        return grad

    def compute_fisher_matrix(self, grads):
        d = grads[0][0].shape[0]
        fisher = torch.zeros((d, d), device=self.device)
        for traj_grads in grads:
            for grad in traj_grads:
                fisher += grad @ grad.T / len(traj_grads)
        fisher /= len(grads)
        fisher += self.lamb * torch.eye(d, device=self.device)
        return fisher

    def compute_value_gradient(self, grads, rewards):
        N = len(grads)
        b = torch.mean(torch.tensor([torch.sum(r) for r in rewards], device=self.device, dtype=torch.float32))
        v_grad = torch.zeros(grads[0][0].shape, device=self.device, dtype=torch.float32)
        for traj_grads, traj_rewards in zip(grads, rewards):
            for t, grad in enumerate(traj_grads):
                discounted_reward = torch.sum(torch.tensor(traj_rewards[t:], device=self.device, dtype=torch.float32))
                v_grad += (discounted_reward - b) * grad / len(traj_grads)
        return v_grad / N

    def compute_eta(self, fisher, v_grad):
        eta = torch.sqrt(self.delta / (v_grad.T @ torch.linalg.inv(fisher) @ v_grad + self.delta))
        return eta.item()

    def train_step(self, theta, trajectories, rewards):
        grads = []
        for traj in trajectories:
            traj_grads = []
            for state, action in traj:
                features = torch.tensor(state, dtype=torch.float32, device=self.device)
                grad = self.compute_log_softmax_grad(theta, features, action)
                traj_grads.append(grad)
            grads.append(traj_grads)

        fisher = self.compute_fisher_matrix(grads)
        v_grad = self.compute_value_gradient(grads, rewards)
        eta = self.compute_eta(fisher, v_grad)

        theta += eta * torch.linalg.inv(fisher) @ v_grad
        return theta

if __name__ == "__main__":
    from call_menu_env import CallMenuEnv
    import utils

    class CallMenuEnvWrapper:
        def __init__(self, env):
            self.env = env

        def reset(self):
            obs, info = self.env.reset()
            return utils.flatten_observation(obs), info["action_mask"]

        def step(self, action):
            obs, reward, done, truncated, info = self.env.step(action)
            return utils.flatten_observation(obs), reward, done, info["action_mask"]
    menu = {
    "number": 1,
    "text": "Welcome to our service.",
    "is_target": False,
    "children": [
        {
            "number": 1,
            "text": "For customer service representative, press 1.",
            "is_target": False,
            "children": [
                {
                    "number": 1,
                    "text": "For technical support, press 1.",
                    "is_target": False,
                    "children": [
                        {
                            "number": 1,
                            "text": "For software related issues, press 1.",
                            "is_target": False,
                            "children": []
                        },
                        {
                            "number": 2,
                            "text": "For hardware related issues, press 2.",
                            "is_target": False,
                            "children": []
                        },
                        {
                            "number": 3,
                            "text": "To speak directly with a technical support representative, press 3.",
                            "is_target": False,
                            "children": []
                        }
                    ]
                },
                {
                    "number": 2,
                    "text": "For billing inquiries, press 2.",
                    "is_target": False,
                    "children": [
                        {
                            "number": 1,
                            "text": "For general billing questions, press 1.",
                            "is_target": False,
                            "children": []
                        },
                        {
                            "number": 2,
                            "text": "For specific invoice queries, press 2.",
                            "is_target": False,
                            "children": []
                        },
                        {
                            "number": 3,
                            "text": "For payment methods, press 3.",
                            "is_target": False,
                            "children": []
                        }
                    ]
                },
                {
                    "number": 3,
                    "text": "For questions about our products, press 3.",
                    "is_target": False,
                    "children": [
                        {
                            "number": 1,
                            "text": "For information about product warranties, press 1.",
                            "is_target": False,
                            "children": []
                        },
                        {
                            "number": 2,
                            "text": "To speak with a product specialist, press 2.",
                            "is_target": False,
                            "children": []
                        },
                        {
                            "number": 3,
                            "text": "For information about product returns, press 3.",
                            "is_target": False,
                            "children": []
                        }
                    ]
                }
            ]
        }
    ]}
    env = CallMenuEnv(menu)
    env = CallMenuEnvWrapper(env)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rbf_feature = RBFSampler(gamma=1, random_state=12345)

    feature_dim = 100
    action_dim = env.env.action_space.n
    agent = NPGAgent(feature_dim, action_dim, device=device)

    theta = torch.randn(feature_dim, action_dim, device=device, dtype=torch.float32)
    num_episodes = 1000
    trajectories = []
    rewards = []

    for episode in range(num_episodes):
        state, action_mask = env.reset()
        done = False

        episode_rewards = []
        episode_trajectory = []
        
    while not done:
        features = utils.extract_features(state, action_dim).astype(np.float32)
        logits = theta.T @ torch.tensor(features, device=device, dtype=torch.float32)
        probs = torch.softmax(logits, dim=0).squeeze().cpu().numpy()
    
        # Mask invalid actions
        probs = probs * action_mask  # Zero out probabilities of invalid actions
        if probs.sum() == 0:  # Avoid divide-by-zero
            probs = action_mask / action_mask.sum()  # Uniformly distribute over valid actions
        else:
            probs = probs / probs.sum()  # Normalize probabilities
    
        action = np.random.choice(action_dim, p=probs)
        state, reward, done, action_mask = env.step(action)
    
        episode_rewards.append(reward)
        episode_trajectory.append((features, action))

        trajectories.append(episode_trajectory)
        rewards.append(episode_rewards)

        trajectories.append(episode_trajectory)
        rewards.append(episode_rewards)

        if len(trajectories) >= 100:
            theta = agent.train_step(theta, trajectories, rewards)
            trajectories = []
            rewards = []

        if episode % 100 == 0:
            avg_reward = np.mean([np.sum(r) for r in rewards])
            print(f"Episode {episode}: Average Reward: {avg_reward}")

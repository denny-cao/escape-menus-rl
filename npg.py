import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from call_menu_env import CallMenuEnv

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

class NPGAgent:
    def __init__(self, observation_space, action_space, lr=1e-3, gamma=0.99, device="cpu"):
        self.gamma = gamma
        self.device = device

        input_dim = observation_space["text_embedding"].shape[0]
        action_dim = action_space.n

        self.policy_network = PolicyNetwork(input_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

    def select_action(self, observation, action_mask):
        text_embedding = torch.tensor(observation["text_embedding"], dtype=torch.float32).to(self.device)

        logits = self.policy_network(text_embedding)
        masked_logits = logits + (1 - torch.tensor(action_mask, dtype=torch.float32).to(self.device)) * -1e9 # Mask logits of invalid actions

        probs = torch.softmax(masked_logits, dim=-1)
        probs_categorical = Categorical(probs)
        action = probs_categorical.sample()
        log_prob = probs_categorical.log_prob(action)

        return action.item(), log_prob

    def update_policy(self, rewards, log_probs):
        discounted_rewards = []
        R = 0
        for reward in reversed(rewards):
            R = reward + self.gamma * R
            discounted_rewards.insert(0, R)

        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(self.device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)

        policy_loss = torch.stack([-log_prob * reward for log_prob, reward in zip(log_probs, discounted_rewards)]).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

if __name__ == "__main__":
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
                                "is_target": True,
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
        ]
    }
    env = CallMenuEnv(menu, max_children=10)
    obs, info = env.reset()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = NPGAgent(env.observation_space, env.action_space, device=device)

    num_episodes = 1000
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False

        rewards = []
        log_probs = []

        while not done:
            action_mask = info["action_mask"]
            action, log_prob = agent.select_action(obs, action_mask)

            print(f"Action: {action}")

            obs, reward, done, truncated, info = env.step(action)

            rewards.append(reward)
            log_probs.append(log_prob)

        agent.update_policy(rewards, log_probs)

        if episode % 100 == 0:
            print(f"Episode {episode}: Total Reward: {sum(rewards)}")


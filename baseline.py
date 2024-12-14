import os
import json
import random
from typing import List, Dict, Optional
import openai
from sample_trajectory import MenuNode

api_key = os.getenv("OPENAI_API_KEY")

class ChatGPTBaselineAgent:
    """
    ChatGPT-based baseline agent to navigate menu trees.
    """
    def __init__(self, json_folder: str):
        openai.api_key = api_key
        self.json_folder = json_folder
        self.menu_tree: Optional[MenuNode] = None
        self.current_node: Optional[MenuNode] = None

    def load_random_menu_tree(self) -> None:
        json_files = [f for f in os.listdir(self.json_folder) if f.endswith('.json')]
        if not json_files:
            raise FileNotFoundError("No JSON files found in the specified folder.")

        selected_file = random.choice(json_files)
        with open(os.path.join(self.json_folder, selected_file), 'r') as file:
            menu_tree_data = json.load(file)

        self.menu_tree = MenuNode.from_dict(menu_tree_data)
        self.current_node = self.menu_tree
        print(f"Loaded menu tree from: {selected_file}")

    def get_best_action(self) -> int:
        if not self.current_node or not self.current_node.children: # No children to evaluate
            raise ValueError("Current node has no children to evaluate.")

        child_texts = [f"{child.number}: {child.text}" for child in self.current_node.children]
        prompt = (
            "You are navigating a phone menu system. Choose the option most likely to connect to a human operator.\n"
            f"Options:\n{chr(10).join(child_texts)}\n"
            "Respond with the number of the best option. (Integer, without leading zeros and no toher text)"
        )

        try:
            response = openai.Completion.create(
                engine="gpt-4o",
                prompt=prompt,
                max_tokens=10,
                temperature=0.7,
            )
            choice = response['choices'][0]['text'].strip()
            return int(choice)
        except Exception as e:
            print(f"Error with OpenAI API call: {e}")
            return random.choice([child.number for child in self.current_node.children])

    def step(self) -> Optional[int]:
        if not self.current_node:
            raise ValueError("Current node is not set. Call `load_random_menu_tree()` first.")

        action = self.get_best_action()
        next_node = self.current_node.get_child_by_number(action)

        if not next_node:
            print(f"Invalid action: {action}")
            return None

        # Update the current node
        self.current_node = next_node

        # Check if the node is the target
        reward = 1 if self.current_node.is_target else 0
        return reward

    def run_episode(self, max_steps: int = 10) -> int:
        self.load_random_menu_tree()
        total_reward = 0

        for _ in range(max_steps):
            reward = self.step()
            if reward is None:
                break

            total_reward += reward

            if len(self.current_node.children) == 0:  # End if no more children
                break

        return total_reward

class RandomPolicyAgent:
    """
    Random policy agent to navigate menu trees.
    """
    def __init__(self, json_folder: str):
        self.json_folder = json_folder
        self.menu_tree: Optional[MenuNode] = None
        self.current_node: Optional[MenuNode] = None

    def load_random_menu_tree(self) -> None:
        json_files = [f for f in os.listdir(self.json_folder) if f.endswith('.json')]
        if not json_files:
            raise FileNotFoundError("No JSON files found in the specified folder.")

        selected_file = random.choice(json_files)
        with open(os.path.join(self.json_folder, selected_file), 'r') as file:
            menu_tree_data = json.load(file)

        self.menu_tree = MenuNode.from_dict(menu_tree_data)
        self.current_node = self.menu_tree
        print(f"Loaded menu tree from: {selected_file}")

    def step(self) -> Optional[int]:
        if not self.current_node or not self.current_node.children:
            return None

        action = random.choice([child.number for child in self.current_node.children])
        next_node = self.current_node.get_child_by_number(action)

        if not next_node:
            return None

        # Update the current node
        self.current_node = next_node

        # Check if the node is the target
        reward = 1 if self.current_node.is_target else 0
        return reward

    def run_episode(self, max_steps: int = 10) -> int:
        self.load_random_menu_tree()
        total_reward = 0

        for _ in range(max_steps):
            reward = self.step()
            if reward is None:
                break

            total_reward += reward

            if len(self.current_node.children) == 0:  # End if no more children
                break

        return total_reward

if __name__ == "__main__":
    # Run ChatGPT-based agent
    agent = ChatGPTBaselineAgent(api_key="your_openai_api_key", json_folder="path_to_json_folder")
    total_rewards = []

    for episode in range(10):
        reward = agent.run_episode()
        total_rewards.append(reward)
        print(f"ChatGPT Agent - Episode {episode + 1}: Total Reward = {reward}")

    print(f"ChatGPT Agent - Average Reward: {sum(total_rewards) / len(total_rewards):.2f}")

    # Run Random policy agent
    random_agent = RandomPolicyAgent(json_folder="path_to_json_folder")
    random_rewards = []

    for episode in range(10):
        reward = random_agent.run_episode()
        random_rewards.append(reward)
        print(f"Random Agent - Episode {episode + 1}: Total Reward = {reward}")

    print(f"Random Agent - Average Reward: {sum(random_rewards) / len(random_rewards):.2f}")











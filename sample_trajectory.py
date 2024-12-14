import json
import os
import random
from typing import List, Tuple, Dict, Optional
import yaml
import torch
from transformers import AutoTokenizer, AutoModel

SPECS_PATH = "specs.yaml"

class MenuNode:
    """
    Represents a node in the menu tree.
    """
    def __init__(self, number: int, text: str, is_target: bool, children: List['MenuNode']):
        self.number = number
        self.text = text
        self.is_target = is_target
        self.children = [MenuNode.from_dict(child) if isinstance(child, dict) else child for child in children]

    @staticmethod
    def from_dict(data: Dict) -> 'MenuNode':
        """
        Create a MenuNode from a dictionary.
        """
        return MenuNode(
            number=data['number'],
            text=data['text'],
            is_target=data['is_target'],
            children=data.get('children', [])
        )
    
    def get_child_by_number(self, number: int) -> Optional['MenuNode']:
        """
        Get a child node by its 'number'.
        """
        for child in self.children:
            if child.number == number:
                return child
        return None

    def get_state_text(self) -> str:
        """
        Get the concatenated text of the children.
        """
        return ' '.join(child.text for child in self.children) if self.children else ''


class TrajectorySampler:
    """
    Samples state-action trajectories from randomly selected JSON menu trees.
    """
    def __init__(self, 
                 json_folder: str, 
                 model_name: str = "sentence-transformers/paraphrase-MiniLM-L6-v2",):
        """
        Initialize the TrajectorySampler with a folder of JSON menu files and a BERT model.
        
        Args:
            json_folder (str): The path to the folder containing the JSON files.
            model_name (str): The name of the pre-trained model to load from HuggingFace.
        """
        # Load MAX_DEPTH_TREE from the specs.yaml file
        with open(SPECS_PATH, 'r') as file:
            specs = yaml.safe_load(file)
            self.max_depth = specs.get("max-depth-tree", 10) 
    
        self.json_folder = json_folder
        self.menu_tree: Optional[MenuNode] = None  # Current active menu tree
        self.current_node: Optional[MenuNode] = None  # Current position in the tree
        self.current_level: int = 0
        # Load the BERT model and tokenizer
        print(f"Loading BERT model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Cache for BERT embeddings to avoid recomputation
        self.embedding_cache = {}

    def load_random_menu_tree(self) -> None:
        """
        Randomly select and load a JSON menu tree from the specified folder.
        """
        json_files = [f for f in os.listdir(self.json_folder) if f.endswith('.json')]
        if not json_files:
            raise FileNotFoundError("No JSON files found in the specified folder.")
        
        selected_file = random.choice(json_files)
        with open(os.path.join(self.json_folder, selected_file), 'r') as file:
            menu_tree_data = json.load(file)
        
        self.menu_tree = MenuNode.from_dict(menu_tree_data)
        self.current_node = self.menu_tree
        print(f"Loaded menu tree from: {selected_file}")
    
    def reset(self) -> torch.Tensor:
        """
        Reset the current trajectory by selecting a new random menu tree and return the initial state.

        Returns:
            torch.Tensor: The BERT embedding for the initial state.
        """
        self.load_random_menu_tree()
        self.current_level = 0
        initial_state_text = self.menu_tree.get_state_text()
        initial_state_embedding = self.get_state_embedding(initial_state_text)
        return initial_state_embedding

    def step(self, action: int) -> Tuple[str, int, bool]:
        """
        Take an action at the current node in the menu tree.

        Args:
            action (int): The action (number) to select the next child.

        Returns:
            Tuple[torch.Tensor, int, bool]: 
                - state: The BERT embedding of the children at the new node.
                - reward: 1 if the selected node is a target, otherwise 0.
                - done: True if there are no more children (end of trajectory), otherwise False.
        """
        if self.current_node is None:
            raise ValueError("Current node is not set. Call `reset()` first.")
        
        # Select the child based on the action
        next_node = self.current_node.get_child_by_number(action)
        
        if next_node is None:
            raise ValueError(f"Invalid action '{action}' for current node.")
        
        # Update the current position in the tree
        self.current_node = next_node
        self.current_level += 1

        # Compute the state, reward, and done flag
        state_text = self.current_node.get_state_text()
        state_embedding = self.get_state_embedding(state_text)
        reward = 1 if self.current_node.is_target else 0
        done = len(self.current_node.children) == 0  # Done if no more children
        return state_embedding, reward, done
    
    def get_state_embedding(self, state_text: str) -> torch.Tensor:
        """
        Get the BERT embedding for the given state text.

        Args:
            state_text (str): The text representing the state.

        Returns:
            torch.Tensor: The BERT embedding of the state.
        """
        if state_text in self.embedding_cache:
            base_embedding = self.embedding_cache[state_text]
        else:
            # Tokenize and encode the text
            inputs = self.tokenizer(state_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            with torch.no_grad():
                outputs = self.model(**inputs)
            base_embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze()
            self.embedding_cache[state_text] = base_embedding
        
        # Create a one-hot vector for the current level
        one_hot_level = torch.zeros(self.max_depth)
        level_index = min(self.current_level, self.max_depth - 1)  # ensure we don't go out of range
        one_hot_level[level_index] = 1.0

        one_hot_level = one_hot_level.to(base_embedding.device, dtype=base_embedding.dtype)
        combined_embedding = torch.cat((base_embedding, one_hot_level), dim=0)

        return combined_embedding
    

    def sample_trajectory(self) -> List[Tuple[torch.Tensor, int, int]]:
        """
        Sample a complete trajectory by taking random actions in the menu tree.

        Returns:
            List[Tuple[torch.Tensor, int, int]]: 
                A list of (state_embedding, action, reward) tuples for each step in the trajectory.
        """
        self.reset()
        trajectory = []
        trajectory.append((self.current_node.get_state_text()))

        for _ in range(self.max_steps):
            if self.current_node is None or len(self.current_node.children) == 0:
                break  # End if no children exist
            
            # Randomly choose a valid action from the available children
            available_actions = [child.number for child in self.current_node.children]
            action = random.choice(available_actions)
            
            # Take the action
            state, reward, done = self.step(action)
            trajectory.append((state, action, reward))
            
            if done:
                break
        
        return trajectory


if __name__ == "__main__":
    sampler = TrajectorySampler("pr_25_br_3_dp_3")
    
    trajectory = sampler.sample_trajectory()
    # print(trajectory)

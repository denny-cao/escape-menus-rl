# tree_generation.py
import json
from typing import List
from generate_data import generate_menu_tree, export_menu_tree_to_json

def generate_sample_tree(depth: int = 3, branching_factor: int = 3, target_number: int = 1) -> None:
    # Generate the tree with the specified depth, branching factor, and target number
    menu_tree = generate_menu_tree(depth=depth, branching_factor=branching_factor, target_number=target_number)
    
    # Export the generated tree to a JSON file for further use
    export_menu_tree_to_json(menu_tree, "sample_menu_tree.json")

if __name__ == "__main__":
    # Example usage
    generate_sample_tree(depth=3, branching_factor=2, target_number=1)
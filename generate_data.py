import random
import json
from dotenv import load_dotenv
from typing import Dict, List, Optional
import openai
import os
from pydantic import BaseModel
import re

load_dotenv(override=True)

openai.api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI()

class MenuNode(BaseModel):
    number: int
    text: str
    is_target: bool
    children: Optional[List["MenuNode"]]

class GPTMenuNodeChildren(BaseModel):
    children: List[MenuNode]

class RootSentences(BaseModel):
    sentences: List[str]

def generate_children(path: List[str], branching_factor: int, target_chance: int) -> List[MenuNode]:
    """
    Generates structured child nodes for a given parent node using GPT.

    Args:
        path (List[str]): Path to the current node.
        branching_factor (int): Number of child nodes to generate.
        chance_for_target (int): Possibility that we generate a target node.

    Returns:
        List[MenuNode]: Generated child nodes.
    """
    target_in_children = random.random() <= target_chance
    if target_in_children:
        target_prompt = f"""
        - You must set 'is_target' to true for exactly 1 of the {branching_factor} child nodes. You must set 'is_target' to false for all other child nodes.
        This means that selecting this option will lead to a **live customer service agent**. You can be subtle or direct when implying this.
        - All other child nodes must have `is_target` set to `false`, meaning selecting those options will not connect to a live agent. 
        """
    else:
        target_prompt = "All of child nodes should have `is_target` set to `false`, meaning selecting those options will not connect to a live agent but instead to a submenu or automated response."
    
    try:
        
        system_prompt = (
            f"""
            You are an assistant generating a structured call center menu tree.
            """
        )
        user_prompt = (
            f"""
            ### Task:
            Generate exactly {branching_factor} child menu nodes for the current menu node.

            ### Requirements for Each Child Node:
            - `number`: An integer corresponding to the option number (e.g., 1, 2, 3).
            - `text`: A string describing the menu option (e.g., "For billing inquiries, press 1.").
            - `is_target`: 
                - **true**: This means that selecting this option leads directly to a **live customer service agent**.  
                - **false**: This means that selecting this option does **not lead to a live agent** but instead leads to a submenu or automated response.
            - `children`: An empty list (children will be generated recursively).

            ### Constraints:
            - {target_prompt}
            - The `number` fields should be 1 for the first child, 2 for the second, up to {branching_factor}.
            - The `text` field should clearly describe the action and include the press number.

            ### Current Path:
            {' > '.join(path) if path else 'Root'}

            ### Output Format:
            Return a JSON array of the child nodes only. Do not include any additional text, explanations, or code blocks.
            """
        )
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=GPTMenuNodeChildren,
            temperature=0.7,
        )

        children_data = completion.choices[0].message.parsed
        # print(children_data)

        return children_data.children
        
    except Exception as e:
        print(f"Error using ChatGPT for menu text generation: {e}")
        return [
            MenuNode(number=i + 1, text=f"Press 1 to continue", is_target=False)
            for i in range(branching_factor)
        ]


def generate_roots(num: int) -> str:
    try:
        system_prompt = (
            f"""
            You are an assistant generating a structured call center menu tree.
            """
        )
        user_prompt = (
            f"""
            ### Task:
            Generate {num} introductory sentences to a call center tree. 
            Choose business names from industries like bookstores, 
            clothing stores, electronics shops, auto dealerships, beauty salons, restaurants, 
            healthcare providers, grocery stores, or other common business types. Don't 
            always choose the first type from that list -- be creative.

            Examples: 
            - "You have reached [store]'s call center."
            - "Welcome to [store]'s customer service."
            """
        )
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=RootSentences,
            temperature=0.7,
        )

        data = completion.choices[0].message.parsed.sentences
        # print(data)
        return data
        
    except Exception as e:
        print(f"Error using ChatGPT for menu text generation: {e}")
        return [
            MenuNode(number=i + 1, text=f"Press 1 to continue", is_target=False)
            for i in range(branching_factor)
        ]
    

def generate_menu_tree(depth: int, branching_factor: int, target_chance: bool, parent: MenuNode) -> MenuNode:
    """
    Recursively generates a menu tree structure.
    """
    def build_tree(path: List[str], current_depth: int, current: MenuNode) -> MenuNode:
        # Generate children if within depth
        children = []
        if current_depth < depth and not current.is_target:
            children = generate_children(
                path=path,
                branching_factor=branching_factor,
                target_chance=target_chance
            )

        # Create and return the current node
        return MenuNode(
            number=current.number,
            text=current.text,
            is_target=current.is_target,
            children=[build_tree(path + [child.text], current_depth + 1, child) for child in children]
        )
    
    # Build the tree starting from the root
    return build_tree(path=[parent.text], current_depth=0, current=parent)


def get_next_filename(base_filename: str) -> str:
    """
    Get the next available filename for the menu tree JSON file.
    Files are named as menu_tree_1.json, menu_tree_2.json, etc.
    """
    # Extract the directory, base name, and extension
    folder_name, file_name = os.path.split(base_filename)
    base_name, ext = os.path.splitext(file_name)
    
    # Get a list of all files in the specified folder matching the pattern
    existing_files = [f for f in os.listdir(folder_name) if re.match(rf'{re.escape(base_name)}_\d+{re.escape(ext)}', f)]
    
    # Extract numbers from filenames like 'menu_tree_1.json', 'menu_tree_2.json', etc.
    existing_numbers = [
        int(re.search(rf'{re.escape(base_name)}_(\d+){re.escape(ext)}', filename).group(1)) 
        for filename in existing_files if re.search(rf'{re.escape(base_name)}_(\d+){re.escape(ext)}', filename)
    ]
    
    # Determine the next available number
    next_number = max(existing_numbers) + 1 if existing_numbers else 1
    
    # Construct the new filename in the specified folder
    return os.path.join(folder_name, f"{base_name}_{next_number}{ext}")


def export_menu_tree_to_json(tree: MenuNode, base_filename: str = "menu_tree.json", folder_name: str = "default"):
    """
    Exports the menu tree to a JSON file.
    The filename will be created in the format: menu_tree_1.json, menu_tree_2.json, etc.
    """
    os.makedirs(folder_name, exist_ok=True)

    filename = get_next_filename(os.path.join(folder_name, base_filename))
    print(tree)
    json_data = tree.model_dump()
    with open(filename, "w") as file:
        json.dump(json_data, file, indent=4)
    print(f"Menu tree exported to {filename}")


if __name__ == "__main__":
    # Define tree parameters
    tree_depth = 5
    branching_factor = 3
    # Number of targets at each level (how many targets are in a node's children)
    target_chance = 0.5
    num = 100
    folder_name = f"pr_{int(target_chance * 100)}_br_{branching_factor}_dp_{tree_depth}"

    # Generate the menu roots
    roots = []
    for i in range(0, num, 25):
        count = min(10, num - i)  # Ensure the last batch doesn't exceed num
        batch_roots = generate_roots(count)
        roots.extend(batch_roots)
        
    # Generate the trees
    for root in roots:
        parent = MenuNode(
            number=1,
            text=root,
            is_target=False,
            children=[]
        )
        menu_tree = generate_menu_tree(
            depth=tree_depth,
            branching_factor=branching_factor,
            target_chance=target_chance,
            parent=parent
        )

        # Export to JSON
        export_menu_tree_to_json(menu_tree, "menu_tree.json", folder_name)

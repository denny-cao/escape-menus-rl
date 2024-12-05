import random
import json
from typing import Dict, List, Optional
import openai
import os
from pydantic import BaseModel

client = openai.OpenAI()

openai.api_key = os.getenv("OPENAI_API_KEY")

class MenuNode(BaseModel):
    number: int
    text: Optional[str]
    is_target: bool
    children: List["MenuNode"] = []

class GPTMenuNodeChildren(BaseModel):
    children: List[MenuNode]

def generate_children(path, branching_factor, target_number):
    """
    Calls OpenAI to generate structured child nodes for a given parent.
    """
    try:
        system_prompt = (
            f"""
            You are generating a structured menu tree for a call center system.

            ### Your Task:
            Generate all child menu nodes for a given parent node in the tree. The information provided includes:
            1. **Current Path**: The sequence of menu options leading to the current node, including the parent node.

            ### Rules for Generation:
            Structure of a node:
            - `number`: The number corresponding to the number that the parent text says to click to get to the child.
            - `text`: The text of the node, corresponding to what someone may hear at one level of a call center menu.
            - `is_target`: A boolean indicating if this node is an agent. If so, the `text` field may be left empty.
            - `children`: Empty list.

            1. If the current node is the **Root**:
            - Generate exactly **one child node**.
            - Assign the 'number' 1 to the child and provide meaningful text.

            2. For all other nodes:
            - Generate exactly **{branching_factor} child nodes**.
            - Each child must:
                - Have a unique number corresponding to the action required to reach it from the parent.
                - Include `text` corresponding to what someone may hear at that level of the menu.
                    Each child's implicit grandchildren in the `text` field should include **{branching_factor} grandchild nodes** 
                    and **{target_number}** of them should lead to speaking to an agent. This can be subtly implied.
                - Specify whether it leads to an agent (`is_target: true`). If so, the `text` field may be left empty.

            EXAMPLE TEXT FOR A NODE:

            [START EXAMPLE TEXT]

            Welcome to [Company Name]'s support center. Please listen carefully to the following options:

            For billing inquiries, press 1. This includes questions about invoices, payment methods, or refund requests.
            For technical support, press 2. Our agents can help you troubleshoot any issues with our products or services.
            To track an order, press 3. You will need your order number or account information handy.
            To speak with a representative, press 4. Please note that hold times may vary.

            [END EXAMPLE TEXT]
            """
        )
        user_prompt = (
            f"""
            Current path: {' > '.join(path) if path else 'Root'}.
            """
        )
        completion = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=GPTMenuNodeChildren,
            temperature=0.7,
        )

        children_data = completion.choices[0].message.parsed
        print(children_data)

        return [MenuNode(**child) for child in children_data]
        
    except Exception as e:
        print(f"Error using ChatGPT for menu text generation: {e}")
        return [
            MenuNode(number=i + 1, text=f"Press 1 to continue", is_target=False)
            for i in range(branching_factor)
        ]

def generate_menu_tree(depth: int, branching_factor: int, target_number: int) -> MenuNode:
    """
    Recursively generates a menu tree structure.
    """
    def build_tree(path: List[str], current_depth: int) -> MenuNode:
        # Determine if the current node is the root
        is_root = current_depth == 0

        # Generate children if within depth
        children = []
        if current_depth < depth:
            children = generate_children(
                path=path,
                branching_factor=branching_factor if not is_root else 1,
                target_number=target_number
            )

        # Create and return the current node
        return MenuNode(
            number=1 if is_root else path[-1].split()[-1],
            text="Root" if is_root else path[-1],
            is_target=False,
            children=[build_tree(path + [child.text], current_depth + 1) for child in children]
        )

    # Build the tree starting from the root
    return build_tree(path=[], current_depth=0)


def export_menu_tree_to_json(tree: MenuNode, filename: str):
    """
    Exports the menu tree to a JSON file.
    """
    with open(filename, "w") as file:
        json.dump(tree.dict(), file, indent=4)
    print(f"Menu tree exported to {filename}")


if __name__ == "__main__":
    # Define tree parameters
    tree_depth = 3
    branching_factor = 3
    # Number of targets at each level (how many targets are in a node's children)
    target_number = 1

    # Generate the menu tree
    menu_tree = generate_menu_tree(
        depth=tree_depth,
        branching_factor=branching_factor,
        target_number=target_number
    )

    # Export to JSON
    export_menu_tree_to_json(menu_tree, "menu_tree.json")

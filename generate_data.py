# generate_data.py

import json

import os
import re
from typing import Optional, List
import openai
from pydantic import BaseModel

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


openai.api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI()

class MenuNode(BaseModel):
    number: str
    text: Optional[str]
    is_target: bool
    children: Optional[List["MenuNode"]]

MenuNode.update_forward_refs()

def generate_children(path: List[str], branching_factor: int, target_number: int) -> List[MenuNode]:
    """
    Generate children menu nodes using the OpenAI API with strict adherence to the rules.
    """

    is_root = (len(path) == 0)
    desired_count = 1 if is_root else branching_factor
    parent_context = "Root" if is_root else " > ".join(path)

    # Simplified, very explicit prompt:
    # We define the structure, the rules, and provide a single example.
    system_prompt = f"""
You are creating a JSON array of menu options (child nodes) for an IVR system of "GreenValley Grocers".

### Rules:
- Output: A JSON array of exactly {desired_count} objects.
- Each object has:
  - "number": a single character from [0-9,*,#]
  - "text": a string starting with "Press <number> for..." or "Press <number> to speak with..."
  - "is_target": boolean
  - "children": an empty array []
- No extra text or commentary outside the JSON array.
- Unique "number" for each child.

### Conditions:
- If at the root (no parent selections):
  - Exactly 1 child.
  - is_target=false
  - Provide a top-level option like: "Press 1 for store information."
- If not root:
  - Exactly {branching_factor} children.
  - Exactly {target_number} of them must have is_target=true.
    - A target node must say: "Press X to speak with a customer service representative."
    - Non-target nodes must not mention "representative."
    - Non-target nodes: "Press X for <relevant info>."
- The "relevant info" should be consistent with the parent context.
- The target node (if any) must say "Press X to speak with a customer service representative."
- Non-target nodes must NOT mention "representative."

### Example (if branching_factor=2, target_number=1):
[
  {{
    "number": "1",
    "text": "Press 1 for today's store hours.",
    "is_target": false,
    "children": []
  }},
  {{
    "number": "9",
    "text": "Press 9 to speak with a customer service representative.",
    "is_target": true,
    "children": []
  }}
]

Current context: {parent_context}
Follow these rules exactly.
"""

    user_prompt = ""

    MAX_RETRIES = 3
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                timeout=30
            )
            raw = response.choices[0].message.content.strip()
            logger.info(f"Attempt {attempt}: Raw API Response: {raw}")

            if not raw:
                raise ValueError("Empty response from OpenAI API.")

            parsed = json.loads(raw)

            # Validate number of children
            if len(parsed) != desired_count:
                raise ValueError(f"Expected {desired_count} children, got {len(parsed)}")

            # Validate structure
            numbers_seen = set()
            target_count = 0
            for child in parsed:
                # Check required fields
                if not all(k in child for k in ["number", "text", "is_target", "children"]):
                    raise ValueError("Child node missing required fields.")
                if not isinstance(child["is_target"], bool):
                    raise ValueError("is_target must be boolean.")
                if not isinstance(child["children"], list):
                    raise ValueError("children must be a list.")
                if not child["number"]:
                    raise ValueError("number must not be empty.")
                if child["number"] in numbers_seen:
                    raise ValueError(f"Duplicate number: {child['number']}")
                numbers_seen.add(child["number"])

                text_lower = child["text"].lower()
                # Check formatting of text
                if not child["text"].startswith("Press "):
                    raise ValueError("Text must start with 'Press '")
                if "representative" in text_lower:
                    # must be target = true
                    if not child["is_target"]:
                        raise ValueError("Mentioned 'representative' but is_target=false.")
                    target_count += 1
                else:
                    # No representative mention -> must be non-target
                    if child["is_target"]:
                        raise ValueError("is_target=true without mentioning representative.")
                
            # Check target_count matches target_number for non-root
            if not is_root and target_count != target_number:
                raise ValueError(f"Expected {target_number} target nodes, got {target_count}")

            return [MenuNode(**c) for c in parsed]

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Attempt {attempt}: {e}")
            if attempt < MAX_RETRIES:
                logger.info("Retrying...")
                continue
            else:
                logger.error("Max retries reached. Using fallback.")
                break
        except openai.error.OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            if attempt < MAX_RETRIES:
                logger.info("Retrying...")
                continue
            else:
                logger.error("Max retries reached. Using fallback.")
                break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            if attempt < MAX_RETRIES:
                logger.info("Retrying...")
                continue
            else:
                logger.error("Max retries reached. Using fallback.")
                break

    # Fallback: produce a minimal correct structure
    logger.info("Using fallback placeholder nodes.")
    count = desired_count
    possible_keys = [str(i) for i in range(10)] + ["*", "#"]
    children = []
    for i in range(count):
        num = possible_keys[i % len(possible_keys)]
        if is_root:
            # one child, non-target
            txt = f"Press {num} for store hours."
            is_t = False
        else:
            # If we need target_number > 0, put one target node
            # e.g. the last child is target if target_number=1
            if i == count - 1 and target_number == 1:
                txt = f"Press {num} to speak with a customer service representative."
                is_t = True
            else:
                txt = f"Press {num} for more information."
                is_t = False
        children.append(MenuNode(number=num, text=txt, is_target=is_t, children=[]))
    return children


def generate_menu_tree(depth: int, branching_factor: int, target_number: int) -> MenuNode:
    def build_tree(path: List[str], current_depth: int) -> MenuNode:
        is_root = (current_depth == 0)
        if is_root:
            current_number = "1"
            current_text = "Welcome to GreenValley Grocers Customer Support."
            children_nodes = generate_children(path, branching_factor=1, target_number=0)
        else:
            last_text = path[-1]
            match = re.search(r"Press (\d+|\*|#)", last_text)
            current_number = match.group(1) if match else "1"
            current_text = last_text
            children_nodes = []
            if current_depth < depth:
                children_nodes = generate_children(path, branching_factor=branching_factor, target_number=target_number)

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
    # Extract the base and file extension
    base_name, ext = os.path.splitext(base_filename)
    # Get a list of all files in the current directory matching the pattern
    existing_files = [f for f in os.listdir('.') if re.match(rf'{re.escape(base_name)}_\d+{re.escape(ext)}', f)]
    
    # Extract numbers from filenames like 'menu_tree_1.json', 'menu_tree_2.json', etc.
    existing_numbers = [
        int(re.search(rf'{re.escape(base_name)}_(\d+){re.escape(ext)}', filename).group(1)) 
        for filename in existing_files if re.search(rf'{re.escape(base_name)}_(\d+){re.escape(ext)}', filename)
    ]
    
    # Determine the next available number
    next_number = max(existing_numbers) + 1 if existing_numbers else 1
    
    # Construct the new filename
    return f"{base_name}_{next_number}{ext}"

def export_menu_tree_to_json(tree: MenuNode, base_filename: str = "menu_tree.json"):
    """
    Exports the menu tree to a JSON file.
    The filename will be created in the format: menu_tree_1.json, menu_tree_2.json, etc.
    """
    filename = get_next_filename(base_filename)
    print(tree)
    json_data = tree.model_dump()
    with open(filename, "w") as file:
        json.dump(json_data, file, indent=4)
    print(f"Menu tree exported to {filename}")


if __name__ == "__main__":
    # Define tree parameters
    tree_depth = 3
    branching_factor = 3
    # Number of targets at each level (how many targets are in a node's children)
    target_chance = 0.5
    num = 20

    # Generate the menu tree
    roots = generate_roots(num)
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
        export_menu_tree_to_json(menu_tree, "menu_tree.json")


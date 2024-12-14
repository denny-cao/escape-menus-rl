import os
import json
from typing import List, Dict

def clean_children(children: List[Dict], max_number: int) -> List[Dict]:
    """
    Recursively removes child nodes whose 'number' is not in the range [1, 2, 3, ..., n].
    It also recursively cleans the children of the child nodes.

    Args:
        children (List[Dict]): The list of child nodes.
    
    Returns:
        List[Dict]: A cleaned list of child nodes.
    """
    # Filter the children to only keep those whose number is in [1, 2, 3, ..., len(children)]
    valid_numbers = set(range(1, max_number + 1))
    cleaned_children = [child for child in children if child.get('number') in valid_numbers]
    
    # Recursively clean the children of each child node
    for child in cleaned_children:
        if 'children' in child and isinstance(child['children'], list):
            child['children'] = clean_children(child['children'], max_number)
    
    return cleaned_children

def clean_menu_tree(menu_tree: Dict, max_number: int) -> Dict:
    """
    Recursively clean the menu tree's children to ensure only valid 'number' indices exist.
    
    Args:
        menu_tree (Dict): The menu tree JSON data.
    
    Returns:
        Dict: The cleaned menu tree.
    """
    if 'children' in menu_tree and isinstance(menu_tree['children'], list):
        menu_tree['children'] = clean_children(menu_tree['children'], max_number)
    
    return menu_tree

def clean_json_files_in_folder(folder_path: str, output_folder: str = None) -> None:
    """
    Iterate over all JSON files in a folder and remove invalid 'number' indices from the 'children' field.
    
    Args:
        folder_path (str): The path to the folder containing the JSON files.
        output_folder (str, optional): The path to the folder where the cleaned files will be saved. 
                                       If None, files will be overwritten in the same folder.
    """
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    max_number = 3
    
    for json_file in json_files:
        file_path = os.path.join(folder_path, json_file)
        
        with open(file_path, 'r') as file:
            try:
                menu_tree = json.load(file)
            except json.JSONDecodeError as e:
                print(f"Error reading {file_path}: {e}")
                continue
        
        # Clean the menu tree recursively
        cleaned_menu_tree = clean_menu_tree(menu_tree, max_number)
        
        # Determine output path (overwrite or save to a new folder)
        output_path = file_path if output_folder is None else os.path.join(output_folder, json_file)
        
        with open(output_path, 'w') as file:
            json.dump(cleaned_menu_tree, file, indent=4)
        
        print(f"Cleaned file saved to: {output_path}")

if __name__ == "__main__":
    # Specify the folder path where JSON files are located
    input_folder = "pr_75_br_3_dp_3"
    output_folder = "pr_75_br_3_dp_3"
    
    clean_json_files_in_folder(input_folder, output_folder)

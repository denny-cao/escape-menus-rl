import random
import json
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

class MenuNode:
    def __init__(self, id, text, children=None, is_target=False):
        self.id = id
        self.text = text
        self.children = children if children is not None else []
        self.is_target = is_target

    def __repr__(self):
        return f"MenuNode(id={self.id}, text={self.text}, children={len(self.children)}, is_target={self.is_target})"

    def to_dict(self):
        return {
            "id": self.id,
            "text": self.text,
            "is_target": self.is_target,
            "children": [child.to_dict() for child in self.children]
        }

class MenuTree:
    def __init__(self, depth, branching_factor):
        self.depth = depth
        self.branching_factor = branching_factor
        self.root = self._generate_menu_tree()

    @staticmethod
    def generate_menu_text(path, sibling_texts, is_target=False, is_root=False):
        if is_root:
            return ""  # Root node has no text
    
        try:
            prompt = (
                f"You are generating a menu item for a call center system in the form of a tree. Pick a random choic e for the company/field and then generate a menu item name for the current node given the context of path and siblings. The menu item should be concise and unique."
                f"Current path: {' > '.join(path) if path else 'Root'}. "
                f"Siblings: {', '.join(sibling_texts) if sibling_texts else 'None'}. "
                f"{'This node will reach human assistance (The goal state of the tree), but we wish to convey this subtlely.' if is_target else ''} "
            )
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=50,
                n=1,
                stop=None,
                temperature=0.7
            )
            return response.choices[0].text.strip()
        except Exception as e:
            print(f"Error using ChatGPT for menu text generation: {e}")
            # Fallback naming conventions
            if is_target:
                return "Assistance Available"
            return f"Option {len(sibling_texts) + 1}"

    def _generate_menu_tree(self):
        target_placed = [False]  # SOURCE: ChatGPT --- Use a mutable object to track target placement
        node_id_counter = 1
        return self._generate_menu_tree_recursive(
            current_depth=0,
            node_id_counter=node_id_counter,
            context=[],
            target_placed=target_placed,
            sibling_texts=[],
            is_root=True
        )

    def _generate_menu_tree_recursive(self, current_depth, node_id_counter, path, target_placed, sibling_texts, is_root):
        is_target = (
            not is_root and  # Targets cannot be root
            not target_placed[0] and  # Only place one target
            (random.random() < 0.5 or current_depth == self.depth - 1)  # Ensure target is placed at some point
        )
        if is_target:
            target_placed[0] = True
    
        # Target nodes must not have children
        num_children = 0 if is_target else random.randint(2, self.branching_factor) if current_depth < self.depth - 1 else 0
        children = []
    
        # Generate unique text for this node
        node_text = self.generate_menu_text(
            path=path,  # Pass the full path to this node
            sibling_texts=sibling_texts,  # Pass the names of siblings already created
            is_target=is_target,
            is_root=is_root
        )
    
        current_path = path + [node_text]
    
        for i in range(num_children):
            child_sibling_texts = [child.text for child in children]
    
            child = self._generate_menu_tree_recursive(
                current_depth + 1,
                node_id_counter + len(children) + 1,
                current_path,
                target_placed,
                child_sibling_texts,
                is_root=False  # Child nodes are never root
            )
            children.append(child)
    
        return MenuNode(
            id=node_id_counter,
            text=node_text,
            children=children,
            is_target=is_target
        )

    def print_tree(self):
        def print_menu_tree(node, indent=0):
            target_marker = " [TARGET]" if node.is_target else ""
            print('    ' * indent + f"Node ID {node.id}: {node.text}{target_marker}")
            for child in node.children:
                print_menu_tree(child, indent + 1)

        print_menu_tree(self.root)

    def export_to_json(self, filename):
        with open(filename, "w") as file:
            json.dump(self.root.to_dict(), file, indent=4)
        print(f"Menu tree exported to {filename}")


if __name__ == "__main__":
    menu_tree = MenuTree(depth=5, branching_factor=3)
    menu_tree.print_tree()
    menu_tree.export_to_json("menu_tree.json")

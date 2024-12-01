import random
import json

class MenuNode:
    def __init__(self, id, text, children=None, is_target=False):
        self.id = id
        self.text = text
        self.children = children if children is not None else []
        self.is_target = is_target

    def __repr__(self):
        return f"MenuNode(id={self.id}, text={self.text}, children={len(self.children)}, is_target={self.is_target})"

    def to_dict(self):
        """
        Converts the MenuNode and its children into a dictionary format for JSON export.
        """
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
    def generate_menu_text(context, sibling_texts, is_target=False, is_root=False):
        if is_root:
            return "Main Menu"
        if is_target:
            return "Target"
        return f"Option"

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

    def _generate_menu_tree_recursive(self, current_depth, node_id_counter, context, target_placed, sibling_texts, is_root):
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

        for i in range(num_children):
            child_context = context + [f"Option {i + 1}"]
            child_sibling_texts = [child.text for child in children]

            child = self._generate_menu_tree_recursive(
                current_depth + 1,
                node_id_counter + len(children) + 1,
                child_context,
                target_placed,
                child_sibling_texts,
                is_root=False  # Child nodes are never root
            )
            children.append(child)

        return MenuNode(
            id=node_id_counter,
            text=self.generate_menu_text(context, sibling_texts, is_target=is_target, is_root=is_root),
            children=children,
            is_target=is_target
        )

    def print_tree(self):
        """
        Prints the tree structure in a readable format.
        """
        def print_menu_tree(node, indent=0):
            target_marker = " [TARGET]" if node.is_target else ""
            print('    ' * indent + f"Node ID {node.id}: {node.text}{target_marker}")
            for child in node.children:
                print_menu_tree(child, indent + 1)

        print_menu_tree(self.root)

    def export_to_json(self, filename):
        """
        Exports the menu tree to a JSON file.
        """
        with open(filename, "w") as file:
            json.dump(self.root.to_dict(), file, indent=4)
        print(f"Menu tree exported to {filename}")


if __name__ == "__main__":
    menu_tree = MenuTree(depth=5, branching_factor=3)
    menu_tree.print_tree()
    menu_tree.export_to_json("menu_tree.json")

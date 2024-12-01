import random

class MenuNode:
    def __init__(self, id, text, children=None, is_target=False):
        self.id = id
        self.text = text
        self.children = children if children is not None else []
        self.is_target = is_target

    def __repr__(self):
        return f"MenuNode(id={self.id}, text={self.text}, children={len(self.children)}, is_target={self.is_target})"

def generate_menu_text(context, sibling_texts, is_target=False, is_root=False):
    if is_root:
        return "Main Menu"
    if is_target:
        return "Target"
    return f"Option"


def generate_menu_tree(depth, branching_factor):
    target_placed = [False]  # SOURCE: ChatGPT. Use a mutable object to track target placement
    node_id_counter = 1
    return _generate_menu_tree_recursive(
        depth,
        branching_factor,
        current_depth=0,
        node_id_counter=node_id_counter,
        context=[],
        target_placed=target_placed,  
        sibling_texts=[],
        is_root=True 
    )

def _generate_menu_tree_recursive(depth, branching_factor, current_depth, node_id_counter, context, target_placed, sibling_texts, is_root):
    is_target = (
        not is_root and # Targets cannot be root
        not target_placed[0] and # Only place one target
        (random.random() < 0.5 or current_depth == depth - 1) # Ensure target is placed at some point
    )
    if is_target:
        target_placed[0] = True

    # Target nodes must not have children
    num_children = 0 if is_target else random.randint(2, branching_factor) if current_depth < depth - 1 else 0
    children = []

    for i in range(num_children):
        child_context = context + [f"Option {i + 1}"]
        child_sibling_texts = [child.text for child in children]

        child = _generate_menu_tree_recursive(
            depth,
            branching_factor,
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
        text=generate_menu_text(context, sibling_texts, is_target=is_target, is_root=is_root),
        children=children,
        is_target=is_target
    )

def print_menu_tree(node, indent=0):
    target_marker = " [TARGET]" if node.is_target else ""
    print('    ' * indent + f"Node ID {node.id}: {node.text}{target_marker}")
    for child in node.children:
        print_menu_tree(child, indent + 1)

if __name__ == "__main__":
    menu_tree = generate_menu_tree(depth=5, branching_factor=3)
    print_menu_tree(menu_tree)


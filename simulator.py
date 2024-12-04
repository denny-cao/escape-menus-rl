import json
from generate_data import MenuNode, MenuTree

class Simulator:
    def __init__(self, menu_tree: MenuTree):
        self.menu_tree = menu_tree
        self.current_node = menu_tree.root
        self.history = []
        self.step = 0
        self.reached_target = False

    def reset(self):
        self.current_node = self.menu_tree.root
        self.history = []
        self.step = 0

    def swap_tree(self, menu_tree: MenuTree):
        self.menu_tree = menu_tree
        self.reset()

    def get_children(self):
        return self.current_node.children

    def step_to_child(self, child_idx):
        if child_idx >= len(self.current_node.children):
            raise ValueError("Invalid child index")
        self.current_node = self.current_node.children[child_idx]
        self.history.append(child_idx)
        self.step += 1
        if self.current_node.is_target:
            self.reached_target = True

    def to_dict(self):
        return {
            "current_node": self.current_node.to_dict(),
            "history": self.history,
            "step": self.step,
            "reached_target": self.reached_target
        }

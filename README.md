# Intro
- This intro code generates sample trees with the right structure. We generate nodes indicating whether it's a target (talking to a human, for example), with the text ('press 2 for x'), and a list of children for giving the tree structure.
- The code allows for customization of the branching factor, as well as the depth, leaving us with a wide family of trees 
- GPT3 is used as the base model for generation. We might experiment with the new Qwen model that just got released. 
- A drawback we have is that sometimes the targets aren't well identified, as seen in the examples generated. 
# BabaIsYouRL
Baba Is you Game / RL solver

## Q-Learning agent

Uses Optimistic Initial Value trick to enforce exploration

Uses standard Value update:

$$
Q(s,a) \leftarrow Q_{t}(s,a) + \alpha \cdot [R + \gamma \, \underset{A}{\text{max }} Q(s, A) - Q(s,a)]
$$

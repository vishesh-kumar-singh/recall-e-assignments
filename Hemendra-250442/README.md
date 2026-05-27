## Analysis

The low exploration agent learned a safe path the fastest because it mostly followed learned actions and avoided random moves.

The high exploration agent explored continuously, causing unstable rewards and slower convergence due to frequent random actions.

The decaying exploration agent ultimately found the most optimal path because it explored the environment initially and gradually shifted toward exploitation as epsilon decreased.

The difference occurs because reinforcement learning requires balancing exploration and exploitation.

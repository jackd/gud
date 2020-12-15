import matplotlib.pyplot as plt

from gud.data import single_graphs

plt.spy(single_graphs["acm"].adjacency, markersize=0.1)
plt.show()

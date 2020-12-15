import matplotlib.pyplot as plt

from gud.anomolies import with_cliques
from gud.data import single_graphs

data = single_graphs["blogcatalog"]
adjacency = data.adjacency

anomolous, _ = with_cliques(adjacency, clique_size=15, num_cliques=10)

fig, (ax0, ax1) = plt.subplots(1, 2)
plt.sca(ax0)
plt.title("Original")
plt.spy(adjacency, markersize=0.1)
plt.sca(ax1)
plt.title("With cliques")
plt.spy(anomolous, markersize=0.1)

plt.show()

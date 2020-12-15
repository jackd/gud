import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
from gud.anomolies import with_attribute_anomolies
from gud.data import single_graphs

data = single_graphs["blogcatalog"]
attrs = data.node_attrs

anomolous, mapping = with_attribute_anomolies(attrs, 50, 150)
src, dst = mapping.T
orig = attrs[src]
final = attrs[dst]

norms = np.linalg.norm(orig.todense(), axis=1)
anomolous_norms = np.linalg.norm(final.todense(), axis=1)

fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
plt.sca(ax0)
plt.title("Original")
plt.spy(attrs, markersize=0.1)
plt.sca(ax1)
plt.title("With attr anomolies")
plt.spy(anomolous, markersize=0.1)
plt.sca(ax2)
plt.title("Norms")
sns.distplot(norms, label="original")
sns.distplot(anomolous_norms, label="anomolous")
plt.legend()
plt.show()

import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_dbscan(label_name, pts_2d, cluster_labels,
				output_stub, dpi=400, marker_size=70):
	"""
	Parameters
	----------
	label_name     : str      # e.g. "library"
	pts_2d         : (N,2)    # t-SNE output
	cluster_labels : (N,)     # DBSCAN cluster ids (−1 = noise)
	output_stub    : str      # "library_DBSCAN" (no extension)
	"""
	fig, ax = plt.subplots(figsize=(6, 5))

	unique = np.unique(cluster_labels)
	# colour map with enough distinct colours
	cmap   = plt.cm.get_cmap("tab20", len(unique))

	for k in unique:
		mask = cluster_labels == k
		colour = "#cccccc" if k == -1 else cmap(k)
		label  = "Noise" if k == -1 else f"C{k}"
		ax.scatter(pts_2d[mask, 0], pts_2d[mask, 1],
				   s=marker_size,
				   c=[colour],
				   alpha=0.85 if k != -1 else 0.4,
				   linewidths=0,
				   label=label)

	ax.set_xlabel("t-SNE dimension 1")
	ax.set_ylabel("t-SNE dimension 2")
	ax.set_title(f"{label_name} — Feature-level clusters (DBSCAN)", pad=10)

	# vertical legend just outside the plot
	ax.legend(loc="upper left",
			  bbox_to_anchor=(1.02, 1),
			  frameon=True,
			  ncol=1,
			  title="Cluster id")

	fig.tight_layout()
	fig.savefig(f"{output_stub}.png", dpi=dpi, bbox_inches="tight")
	fig.savefig(f"{output_stub}.pdf",            bbox_inches="tight")  # vector copy
	plt.close(fig)
	print(f"✓ saved {output_stub}.png / .pdf  (dpi={dpi})")

def collapse_token_dim(w):
	"""
	w  : ndarray  shape (1, 768, T)
	-> : ndarray  shape (768,)          (T removed)
	"""
	w = w.squeeze(0)          # (768, T)
	return w.mean(axis=1)     # (768,)  – same length for all T
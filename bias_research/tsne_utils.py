from test_dict import test_dict
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

plt.rcParams.update({
	"font.size":        12,   # base size
	"axes.titlesize":   14,
	"axes.labelsize":   12,
	"xtick.labelsize":  10,
	"ytick.labelsize":  10,
	"legend.fontsize":  11,
})

def normalize_embeddings(embeddings):
	"""
	Normalize embeddings to zero mean and unit variance.
	Args:
		embeddings: Numpy array of shape (num_samples, embedding_dim)
	Returns:
		normalized_embeddings: Normalized embeddings
	"""
	scaler = StandardScaler()
	normalized_embeddings = scaler.fit_transform(embeddings)
	return normalized_embeddings


def reduce_dimensionality(embeddings, n_components=50):
	"""
	Reduce dimensionality using PCA.
	Args:
		embeddings: Numpy array of shape (num_samples, embedding_dim)
		n_components: Number of dimensions to retain
	Returns:
		reduced_embeddings: Dimensionality-reduced embeddings
	"""
	pca = PCA(n_components=n_components, random_state=42)
	reduced_embeddings = pca.fit_transform(embeddings)
	return reduced_embeddings


def run_tsne(embeddings, n_components=2, perplexity=30, learning_rate=200, n_iter=1000):
	"""
	Perform t-SNE on embeddings.
	Args:
		embeddings: Numpy array of shape (num_samples, embedding_dim)
		n_components: Number of dimensions for t-SNE (default: 2)
		perplexity: Perplexity parameter for t-SNE (default: 30)
		learning_rate: Learning rate for t-SNE (default: 200)
		n_iter: Number of iterations for optimization (default: 1000)
	Returns:
		tsne_results: t-SNE reduced embeddings (num_samples, n_components)
	"""
	tsne = TSNE(
		n_components=n_components,
		perplexity=perplexity,
		learning_rate=learning_rate,
		n_iter=n_iter,
		random_state=42,
	)
	tsne_results = tsne.fit_transform(embeddings)
	return tsne_results

def plot_tsne(
	title,               # plot title
	tsne_results,        # (N, 2)
	labels,              # (N,)
	class_names,         # list[str] same order as unique labels
	output_path=None,    # "foo.png"  (".pdf" / ".svg" auto-added)
	dpi=400,             # output resolution for PNG
	s=70,              # marker size
	ncol=2,
):
	"""
	High-quality t-SNE scatter plot.

	Notes
	-----
	* Saves both {output_path}.png (raster, high-DPI) and
	  {output_path}.pdf (vector) so the paper looks crisp.
	* Close the figure after saving to avoid memory leaks in loops.
	"""
	fig, ax = plt.subplots(figsize=(6, 5)) 
	for idx, name in enumerate(class_names):
		pts = tsne_results[labels == idx]
		ax.scatter(
			pts[:, 0],
			pts[:, 1],
			label=name,
			alpha=0.7,
			s=s,             # marker size
			linewidths=0,     # no edge
		)

	ax.set_xlabel("t-SNE dimension 1")
	ax.set_ylabel("t-SNE dimension 2")
	ax.set_title(title, pad=10)
	ax.legend(frameon=True, ncol=ncol)          # tidy legend
	ax.grid(False)

	fig.tight_layout()

	if output_path:                         # e.g.  "gender_science_BEiT"
		root, ext = os.path.splitext(output_path)
		ext = ext.lower() if ext else ".png"
		if ext != ".png":
			root = output_path              # user gave ".pdf" or ".svg"

		fig.savefig(f"{root}.png", dpi=dpi, bbox_inches="tight")

	plt.close(fig)
	

indice_set = ['Gender-Career','Gender-Science']

def get_plot(models):
	model_name = list(models.keys())
	model_name = [name.lower() for name in model_name]
	for index in indice_set:
		test_name = index
		out_dir = test_name.lower()
		file1 = "backbone_embeddings/"+test_dict[test_name][0] +"_"
		file2 = "backbone_embeddings/"+test_dict[test_name][1]+"_"
		file3 = "backbone_embeddings/"+test_dict[test_name][2]+"_"
		file4 = "backbone_embeddings/"+test_dict[test_name][3]+"_"
		if not os.path.exists(out_dir):
			os.makedirs(out_dir)
		for model in models:
			df1 = pd.read_csv(file1+model+"_None.csv", index_col=0).drop(columns=["img"])
			df2 = pd.read_csv(file2+model+"_None.csv",index_col=0).drop(columns=["img"])
			df3 = pd.read_csv(file3+model+"_None.csv",index_col=0).drop(columns=["img"])
			df4 = pd.read_csv(file4+model+"_None.csv",index_col=0).drop(columns=["img"])
			x = []
			y = []
			for row in df1.iterrows():
				x.append(np.array(row[1].values))
				y.append(0)
			for row in df2.iterrows():
				x.append(np.array(row[1].values))
				y.append(1)
			for row in df3.iterrows():
				x.append(np.array(row[1].values))
				y.append(2)
			for row in df4.iterrows():
				x.append(np.array(row[1].values))
				y.append(3)
			x = np.array(x)
			y = np.array(y)
			x = normalize_embeddings(x)

			# 3. Reduce dimensionality with PCA
			x = reduce_dimensionality(x, n_components=50)
			tsne_results = run_tsne(x)
			class_names = [test_dict[test_name][0].split("_")[-1], 
						   test_dict[test_name][1].split("_")[-1], 
						   test_dict[test_name][2].split("_")[-1], 
						   test_dict[test_name][3].split("_")[-1]]
			plot_tsne(f"{model}(Backbone) — {test_name}", tsne_results, y, class_names, \
				output_path=f"{out_dir}/{test_name}_{model}_Backbone.png")
			
		file1 = "logits_embeddings/"+test_dict[test_name][0] +"_"
		file2 = "logits_embeddings/"+test_dict[test_name][1]+"_"
		file3 = "logits_embeddings/"+test_dict[test_name][2]+"_"
		file4 = "logits_embeddings/"+test_dict[test_name][3]+"_"
		for model in models:
			df1 = pd.read_csv(file1+model+"_None.csv", index_col=0).drop(columns=["img"])
			df2 = pd.read_csv(file2+model+"_None.csv",index_col=0).drop(columns=["img"])
			df3 = pd.read_csv(file3+model+"_None.csv",index_col=0).drop(columns=["img"])
			df4 = pd.read_csv(file4+model+"_None.csv",index_col=0).drop(columns=["img"])
			x = []
			y = []
			for row in df1.iterrows():
				x.append(np.array(row[1].values))
				y.append(0)
			for row in df2.iterrows():
				x.append(np.array(row[1].values))
				y.append(1)
			for row in df3.iterrows():
				x.append(np.array(row[1].values))
				y.append(2)
			for row in df4.iterrows():
				x.append(np.array(row[1].values))
				y.append(3)
			x = np.array(x)
			y = np.array(y)
			x = normalize_embeddings(x)

			# 3. Reduce dimensionality with PCA
			print(f"PCA input shape: {x.shape} for model: {model}, test: {test_name}")
			x = reduce_dimensionality(x, n_components=50)
			tsne_results = run_tsne(x)
			plot_tsne(f"{model}(Logits) — {test_name}", tsne_results, y, class_names, \
				output_path=f"{out_dir}/{test_name}_{model}_logits.png")

def get_weight(activations, grads):
	sum_activations = np.sum(activations, axis=(2, 3))
	eps = 1e-7
	weights = grads * activations / \
		(sum_activations[:, :, None, None] + eps)
	weights = weights.sum(axis=(2, 3))
	return weights
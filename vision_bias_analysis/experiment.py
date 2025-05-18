from typing import List
import pandas as pd
import numpy as np
import os
import gc
import pickle
from PIL import Image
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import Counter
from mmpretrain import ImageClassificationInferencer, get_model
from imagenet_label import IMAGENET_CATEGORIES
from gradcam_utils import generate_gradcam, return_img
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from ieat.api import test_all
from tsne_utils import get_plot, run_tsne, reduce_dimensionality, normalize_embeddings, get_weight, plot_tsne
from tqdm import tqdm
from activation_ratio import get_activation_ratio

def ieat_run():
	'''
	test_all dictionary:
	{
		'model_name': [checkpoint, backbone]
	}
	'''
	results_backbone = test_all(
	{
			'beit': ["beit-base-p16_beitv2-in21k-pre_3rdparty_in1k", True],
			'maskfeat': ["vit-base-p16_maskfeat-pre_8xb256-coslr-100e_in1k", True],
			'mae':["vit-base-p16_mae-1600e-pre_8xb128-coslr-100e_in1k", True],
			'milan':["vit-base-p16_milan-pre_8xb2048-linear-coslr-100e_in1k", True],
			'simmm':["swin-base-w6_simmim-800e-pre_8xb256-coslr-100e_in1k-192px", True],
			'moco': ["vit-base-p16_mocov3-pre_8xb64-coslr-150e_in1k", True],
			'dino':['facebookresearch/dinov2', 'dinov2_vitb14_lc', True],
			'deit':["deit-base-distilled_3rdparty_in1k", True],
			'vit': ["vit-base-p16_in21k-pre_3rdparty_in1k-384px", True],
			'swin':["swin-base_in21k-pre-3rdparty_in1k", True],
	},
	)
	results_df = pd.DataFrame(results_backbone).transpose()
	results_df.columns = ["X", "Y", "A", "B", "d", "x_ab", "y_ab", "p", "n_t", "n_a"]
	for c in results_df.columns[:4]:
		results_df[c] = results_df[c].str.split("/").str[-1]
	results_df["sig"] = ""
	for l in [0.10, 0.05, 0.01]:
		results_df.sig[results_df.p < l] += "*"
	results_df.to_csv("ieat_results_backbone.csv")
	results_logits = test_all(
	{
			'beit': ["beit-base-p16_beitv2-in21k-pre_3rdparty_in1k", False],
			'maskfeat': ["vit-base-p16_maskfeat-pre_8xb256-coslr-100e_in1k", False],
			'mae':["vit-base-p16_mae-1600e-pre_8xb128-coslr-100e_in1k", False],
			'milan':["vit-base-p16_milan-pre_8xb2048-linear-coslr-100e_in1k", False],
			'simmm':["swin-base-w6_simmim-800e-pre_8xb256-coslr-100e_in1k-192px", False],
			'moco': ["vit-base-p16_mocov3-pre_8xb64-coslr-150e_in1k", False],
			'dino':['facebookresearch/dinov2', 'dinov2_vitb14_lc', False],
			'deit':["deit-base-distilled_3rdparty_in1k", False],
			'vit': ["vit-base-p16_in21k-pre_3rdparty_in1k-384px", False],
			'swin':["swin-base_in21k-pre-3rdparty_in1k", False],
	},
	)
	results_df = pd.DataFrame(results_logits).transpose()
	results_df.columns = ["X", "Y", "A", "B", "d", "x_ab", "y_ab", "p", "n_t", "n_a"]
	for c in results_df.columns[:4]:
		results_df[c] = results_df[c].str.split("/").str[-1]
	results_df["sig"] = ""
	for l in [0.10, 0.05, 0.01]:
		results_df.sig[results_df.p < l] += "*"
	results_df.to_csv("ieat_results_logits.csv")
 
def gradCam_run(models: List, labels: List):
	'''
	models: contains the model name and checkpoint
	labels: contains the label name and index in the imagenet dataset (you can refer to imagenet_label.py for the index)
	
	save gradcam images and pkl files containing activation, gradients and grey scale images for a specific label per model
	'''
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	for model_name, ck in models.items():
		if not os.path.exists(model_name):
			os.makedirs(model_name)
	
		if model_name == "dino":
			model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_lc', pretrained = True).eval()
			for label_name, label_index in tqdm(labels.items(), desc=f"{model_name} Processing labels"):
				targets = [ClassifierOutputTarget(label_index)]
				for b in ["male", "female"]:
					if not os.path.exists(os.path.join(model_name, b)):
						os.makedirs(os.path.join(model_name, b))
					save_dir = os.path.join(model_name, b)
					image_size = (112 * 5, 112 * 8)
					exp = Image.new('RGB', image_size)
					images = []
					gradcams = {"grad": {}, "act": {}, "grey": {}}
					#if file exists skip
					if os.path.exists(f"{save_dir}/{model_name}_{label_name}_{b}.jpg") and os.path.exists(f"{save_dir}/{model_name}_{label_name}_{b}.pkl"):
						continue
					for path in os.listdir("data/experiments/gender/" + b):
						img, grey, act, grads=generate_gradcam(model, "data/experiments/gender/" + b + "/" + path, "xgradcam", targets)
						gradcams["grad"][path] = grads
						gradcams["act"][path] = act	
						gradcams["grey"][path] = grey
						images.append(img)
						gc.collect()
					for row in range(8):
						for col in range(5):
							offset = (112 * col, 112 * row)
							idx = row * 5 + col
							box = (offset[0], offset[1], offset[0] + 112, offset[1] + 112)
							exp.paste(Image.fromarray(images[idx]).resize((112, 112)), box)
					exp.save(f"{save_dir}/{model_name}_{label_name}_{b}.jpg")
					with open(f"{save_dir}/{model_name}_{label_name}_{b}.pkl", 'wb') as f:
						pickle.dump(gradcams, f)
				for b in ["white-male", "white-female", "black-male", "black-female"]:
					if not os.path.exists(os.path.join(model_name, b)):
						os.makedirs(os.path.join(model_name, b))
					save_dir = os.path.join(model_name, b)
					image_size = (112 * 5, 112 * 4)
					exp = Image.new('RGB', image_size)
					images = []
					gradcams = {"grad": {}, "act": {}, "grey": {}}
					if os.path.exists(f"{save_dir}/{model_name}_{label_name}_{b}.jpg") and os.path.exists(f"{save_dir}/{model_name}_{label_name}_{b}.pkl"):
						continue
					for path in os.listdir("data/experiments/intersectional/" + b):
						img, grey, act, grads=generate_gradcam(model, "data/experiments/intersectional/" + b + "/" + path, "xgradcam", targets)
						gradcams["grad"][path] = grads
						gradcams["act"][path] = act	
						gradcams["grey"][path] = grey
						images.append(img)
						gc.collect()
					for row in range(4):
						for col in range(5):
							offset = (112 * col, 112 * row)
							idx = row * 5 + col
							box = (offset[0], offset[1], offset[0] + 112, offset[1] + 112)
							exp.paste(Image.fromarray(images[idx]).resize((112, 112)), box)
					exp.save(f"{save_dir}/{model_name}_{label_name}_{b}.jpg")
					with open(f"{save_dir}/{model_name}_{label_name}_{b}.pkl", 'wb') as f:
						pickle.dump(gradcams, f)
		else:
			model = get_model(ck[0], pretrained=True)
			inferencer = ImageClassificationInferencer(ck[0], pretrained=True, device =device)
			cfg = inferencer.config
			for label_name, label_index in tqdm(labels.items(), desc=f"{model_name} Processing labels"):
				targets = [ClassifierOutputTarget(label_index)]
				for b in ["male", "female"]:
					if not os.path.exists(os.path.join(model_name, b)):
						os.makedirs(os.path.join(model_name, b))
					save_dir = os.path.join(model_name, b)
					image_size = (112 * 5, 112 * 8)
					exp = Image.new('RGB', image_size)
					images = []
					gradcams = {"grad": {}, "act": {}, "grey": {}}
					if os.path.exists(f"{save_dir}/{model_name}_{label_name}_{b}.jpg") and os.path.exists(f"{save_dir}/{model_name}_{label_name}_{b}.pkl"):
						continue
					for path in os.listdir("data/experiments/gender/" + b):
						if model_name == "milan":
							img, grey, act, grads = return_img(model, cfg, "data/experiments/gender/" + b + "/" + path,
															"XGradCAM", targets, milan=True)
						elif model_name == "beit":
							img, grey, act, grads = return_img(model, cfg, "data/experiments/gender/" + b + "/" + path,
															"XGradCAM", targets, beit=True)
						else:
							img, grey, act, grads = return_img(model, cfg, "data/experiments/gender/" + b + "/" + path,
															"XGradCAM", targets)
						gradcams["grad"][path] = grads
						gradcams["act"][path] = act
						gradcams["grey"][path] = grey
						images.append(img)
						gc.collect()
					for row in range(8):
						for col in range(5):
							offset = (112 * col, 112 * row)
							idx = row * 5 + col
							box = (offset[0], offset[1], offset[0] + 112, offset[1] + 112)  # Create the box tuple
							exp.paste(Image.fromarray(images[idx]).resize((112, 112)), box)

					exp.save(f"{save_dir}/{model_name}_{label_name}_{b}.jpg")
					with open(f"{save_dir}/{model_name}_{label_name}_{b}.pkl", 'wb') as f:
						pickle.dump(gradcams, f, protocol=pickle.HIGHEST_PROTOCOL)
						f.flush()
						os.fsync(f.fileno()) # Save gradcams as pickle

				for b in ["white-male", "white-female", "black-male", "black-female"]:
					if not os.path.exists(os.path.join(model_name, b)):
						os.makedirs(os.path.join(model_name, b))
					save_dir = os.path.join(model_name, b)
					image_size = (112 * 5, 112 * 4)
					exp = Image.new('RGB', image_size)
					images = []
					gradcams = {"grad": {}, "act": {}, "grey": {}}
					if os.path.exists(f"{save_dir}/{model_name}_{label_name}_{b}.jpg") and os.path.exists(f"{save_dir}/{model_name}_{label_name}_{b}.pkl"):
						continue
					for path in os.listdir("data/experiments/intersectional/" + b):
						if model_name == "milan":
							img, grey, act, grads = return_img(model, cfg, "data/experiments/intersectional/" + b + "/" + path,
															"XGradCAM", targets, milan=True)
						elif model_name == "beit":
							img, grey, act, grads = return_img(model, cfg, "data/experiments/intersectional/" + b + "/" + path,
															"XGradCAM", targets, beit=True)
						else:
							img, grey, act, grads = return_img(model, cfg, "data/experiments/intersectional/" + b + "/" + path,
															"XGradCAM", targets)
						gradcams["grad"][path] = grads
						gradcams["act"][path] = act
						gradcams["grey"][path] = grey
						images.append(img)
						gc.collect()
					for row in range(4):
						for col in range(5):
							offset = (112 * col, 112 * row)
							idx = row * 5 + col
							box = (offset[0], offset[1], offset[0] + 112, offset[1] + 112)  # Create the box tuple
							exp.paste(Image.fromarray(images[idx]).resize((112, 112)), box)
					exp.save(f"{save_dir}/{model_name}_{label_name}_{b}.jpg")
					with open(f"{save_dir}/{model_name}_{label_name}_{b}.pkl", 'wb') as f:
						pickle.dump(gradcams, f, protocol=pickle.HIGHEST_PROTOCOL)
						f.flush()
						os.fsync(f.fileno()) # Save gradcams as pickle
	  
def tsne_activation_weights(models, labels):
	'''
	models: contains the model name and checkpoint
 	labels: contains the label name and index in the imagenet dataset (you can refer to imagenet_label.py for the index)
  
  	generate t-SNE visualization for activation weights to observe activation similarity
	'''
	if not os.path.exists("activation_weights"):
		os.makedirs("activation_weights")
	model_name = list(models.keys())
	model_name = [name.lower() for name in model_name]
	total = {}
	for label,_ in labels.items():
		indices=[]
		total[label] = {}
		for key in model_name:
				total[label][key] = {'weights':[]}
				weights = []
				#greys = []
				for dir in os.listdir(f"{key}"):
						if dir == "grad_cam_images" or dir == "gradcam_images":
							continue
						try:
							data = pickle.load(open(f"{key}/{dir}/{key}_{label}_{dir}.pkl", "rb"))
						except:
							print(f"Error loading {key}/{dir}/{key}_{label}_{dir}.pkl")
							break
						grads = data["grad"]
						acts = data["act"]
						for img in grads.keys():
							indices.append(img)
							activations = acts[img][0].numpy()
							grad = grads[img][0].numpy()
							weight = get_weight(activations, grad) # Resulting shape: (160, 768)
							weights.append(weight)
							#greys.append(grey)
				total[label][key]['weights'] = weights
				#[label][key]['grey'] = greys
	for num in [20]:
		x = []
		y = []
		count = 0
		for label, _ in labels.items():
			x = []
			y = []
			count = 0
			for model in model_name:
				x.extend(np.array(total[label][model]["weights"]))
				y.extend([count for i in range(len(total[label][model]["weights"]))])
				count += 1
			x1=np.array(x[:1120])
			x2=np.array(x[1120:])
			x1 = x1.reshape(-1, 768)
			x2 = x2.reshape(-1, 1024)
			x1 = normalize_embeddings(x1)
			x2 = normalize_embeddings(x2)
			x1 = reduce_dimensionality(x1, n_components=num)
			x2 = reduce_dimensionality(x2, n_components=num)
			x = np.vstack((x1,x2))
			y = np.array(y)
			tsne_results = run_tsne(x)
			plot_tsne(f"t-SNE visualization for activation weights on {label}", \
				tsne_results, y, models, output_path=f"activation_weights/activation_weights_{label}.png", s=30, ncol=1)

def dbscan_plot(models, labels):
	'''
	models: contains the model name and checkpoint
	labels: contains the label name and index in the imagenet dataset (you can refer to imagenet_label.py for the index)
	activation similarity clustering using DBSCAN
	'''
	if not os.path.exists("DBSCAN"):
		os.makedirs("DBSCAN")
	model_name = list(models.keys())
	model_name = [name.lower() for name in model_name]
	for label, _ in labels.items():
		dataset = []
		weights = []
		indices=[]
		for key in model_name:
				for dir in os.listdir(f"{key}"):
					data = pickle.load(open(f"{key}/{dir}/{key}_{label}_{dir}.pkl", "rb"))
					grads = data["grad"]
					acts = data["act"]
					for img in grads.keys():
						indices.append(img)
						activations = acts[img][0].numpy()
						grad = grads[img][0].numpy()
						weight = get_weight(activations, grad) # Resulting shape: (160, 768)
						weights.append(weight)
				dataset.append(weights)
		vit_base = np.array(weights[:1120])
		swin_base = np.array(weights[1120:])

		reshaped_vit = vit_base.reshape(-1, 768)  # Shape: (10 * 160, 768)
		reshaped_swin = swin_base.reshape(-1, 1024)
		scaler = StandardScaler()
		reshaped_vit = scaler.fit_transform(reshaped_vit)
		reshaped_swin = scaler.fit_transform(reshaped_swin)
		# Apply PCA
		pca = PCA(n_components=20, random_state=42)  # Reduce to 50 dimensions
		reshaped_vit = pca.fit_transform(reshaped_vit)
		reshaped_swin = pca.fit_transform(reshaped_swin)  # Shape: (1600, 50)
		data = np.vstack((reshaped_vit, reshaped_swin))
		dbscan = DBSCAN(eps=0.3899999999999999, min_samples=14, metric='cosine')
		labels = dbscan.fit_predict(data)
		tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
		data_2d = tsne.fit_transform(data)
		fig, ax = plt.subplots(figsize=(6, 5))          # same canvas
		marker_size = 70                                 # same as plot_tsne
		unique_lbl  = np.unique(labels)
		cmap        = plt.cm.get_cmap("viridis", len(unique_lbl))

		for l in unique_lbl:
			pts = data_2d[labels == l]
			col = "#cccccc" if l == -1 else cmap(l)      # grey for noise
			lab = "Noise"   if l == -1 else f"Cluster{l+1}"
			ax.scatter(pts[:, 0], pts[:, 1],
					s=marker_size,
					c=[col],
					alpha=0.85 if l != -1 else 0.4,
					linewidths=0,
					label=lab)

		ax.set_title(f"{label}—Feature-level clusters(iEAT data)", pad=10)
		ax.legend(loc="upper left",
				bbox_to_anchor=(1.02, 1),
				frameon=True,
				ncol=1,
				title="Cluster id")
		fig.tight_layout()
		fig.savefig(f"DBSCAN/{label}_DBSCAN.png", dpi=400, bbox_inches="tight")
		plt.close(fig)
		
		count = 0
		plot_labels={}
		for i in range(0, len(labels), 160):
			target = labels[i:i+160]
			print(Counter(target), model_name[count])
			plot_labels[model_name[count]] = Counter(target)
			count+=1
		pd.DataFrame(plot_labels).T.to_json(f"DBSCAN/{label}_DBSCAN_labels.json")
	  
def main():
	# iEAT results section
	# ieat_run()
 
	# # gradCam results section(save act/grad too)
	models = {
		'beit': ["beit-base-p16_beitv2-in21k-pre_3rdparty_in1k"],
		'maskfeat': ["vit-base-p16_maskfeat-pre_8xb256-coslr-100e_in1k"],
		'milan':["vit-base-p16_milan-pre_8xb2048-linear-coslr-100e_in1k"],
		'moco': ["vit-base-p16_mocov3-pre_8xb64-coslr-150e_in1k"],
		'dino':['facebookresearch/dinov2', 'dinov2_vitb14_lc'],
		'deit':["deit-base-distilled_3rdparty_in1k"],
		'vit': ["vit-base-p16_in21k-pre_3rdparty_in1k-384px"],
		'swin':["swin-base_in21k-pre-3rdparty_in1k"],
	}

	labels = {
		"library" : 624,
		"labcoat" : 617,
		"stethoscope" : 823,
		"jeans" : 608,
		"cardigan" : 474,
		"miniskirt" : 655,
		"academicgown" : 400,
		"pajamas" : 697,
		"beaker" : 438,
		"oscilloscope" : 688,
		"suit" : 834,
		"notePC" : 681,
		"sunscreen" : 838,
		"wig" : 903,
		"lipstick" :629,
		"bra" : 459,
		"windsor" : 906,
		"desk" : 526,
		}
	
	# gradCam_run(models, labels)
 
	# # t-SNE results section
	# #e.g models=["beit","maskfeat","mae", "milan","simmm", "moco","dino","deit","vit","swin"]
	# get_plot(models)
	
	# # t_sne activation weights
	# tsne_activation_weights(models, labels)
	
   	# #DBSCAN clustering
	# dbscan_plot(models, labels)
 
	#activation ratio
	get_activation_ratio(models, labels)	

	
if __name__ == "__main__":
	main()


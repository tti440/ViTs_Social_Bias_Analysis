import numpy as np
import os
import pandas as pd
import torch
import json
import pickle
import copy

def get_weight(activations, grads):
	sum_activations = np.sum(activations, axis=(2, 3))
	eps = 1e-7
	weights = grads * activations / \
			(sum_activations[:, :, None, None] + eps)
	weights = weights.sum(axis=(2, 3))
	return weights

def get_total(labels, models):
	total = {}
	for label,_ in labels.items():
		indices=[]
		total[label] = {}
		for key, _ in models.items():
			total[label][key] = {}
			for dir in os.listdir(f"D:/{key}"):
				if dir == "grad_cam_images" or dir == "gradcam_images":
					continue
				jsonfile = json.load(open(f"D:/{key}/{dir}/{key}_{label}_{dir}.json", "r"))
				grey = jsonfile['grey']
				total[label][key][dir] = grey
	return total

def get_activation_ratio(models, labels):
	act_scores={}
	model_names = [model for model in models.keys()]
	if not os.path.exists("activation_ratio"):
		os.makedirs("activation_ratio")
	for model in model_names:
		size = None
		if model == "vit":
			from mask_index_384 import mask_index_bm, mask_index_bf, mask_index_wm, mask_index_wf, mask_index_male, mask_index_female
			size = 384
		else:
			from mask_index_224 import mask_index_bm, mask_index_bf, mask_index_wm, mask_index_wf, mask_index_male, mask_index_female
			size = 224
		mask_indices = {
		"black-male" : mask_index_bm,
		"black-female" : mask_index_bf,
		"white-male" : mask_index_wm,
		"white-female" : mask_index_wf,
		"male" : mask_index_male,
		"female" : mask_index_female,
		}
		masks = pickle.load(open(f"mask_{size}.pkl", "rb"))
		act_ratio = {}
		for label in list(labels.keys()):
			act_ratio[label] = {}
			for dir in os.listdir(f"{model}"):
				tmp = pickle.load(open(f"{model}/{dir}/{model}_{label}_{dir}.pkl", "rb"))
				data = tmp['grey']
				count = 0
				score_face = 0
				score_body = 0	
				for key, value in data.items():
					activation_map = torch.tensor(value)
					mask_index = mask_indices[dir][key]
					face_mask = masks[dir][key][0][0][mask_index[0]]
					human_mask = masks[dir][key][0][0][mask_index[1]]
					body_mask = (human_mask & face_mask ^ human_mask)
					threshold_value = np.percentile(activation_map, 80)
					# Step 3: Apply threshold to the activation map
					activated_pixels = activation_map >= threshold_value
					total_activation = activated_pixels.count_nonzero()
					activated_in_face = (activated_pixels & face_mask.unsqueeze(0)).count_nonzero()
					activated_in_body = (activated_pixels & body_mask.unsqueeze(0)).count_nonzero()
					count += 1
					score_face += activated_in_face/total_activation
					score_body += activated_in_body/total_activation
				act_ratio[label][dir] = [score_face/count, score_body/count]
		act_scores[model] = act_ratio
	with open(f"activation_ratio/activation_ratio.pkl", "wb") as f:
		pickle.dump(act_scores, f)
	avg_total = {}
	dir_len = len(os.listdir(f"{model}"))
	label_size = len(labels)
	for model in model_names:
		avg_total[model] = {'face':0, 'body':0,  "label_avg" : {}, "dir_avg" : {}}
		data = copy.deepcopy(act_scores[model])
		for label in labels.keys():
			for dir in os.listdir(f"{model}"):
				score_face = act_scores[model][label][dir][0]
				score_body = act_scores[model][label][dir][1]
				avg_total[model]["face"] += score_face
				avg_total[model]["body"] += score_body
				if label not in avg_total[model]["label_avg"]:
					avg_total[model]["label_avg"][label] = [score_face, score_body]
				else:
					avg_total[model]["label_avg"][label][0] += score_face
					avg_total[model]["label_avg"][label][1] += score_body
				if dir not in avg_total[model]["dir_avg"]:
					avg_total[model]["dir_avg"][dir] = [score_face, score_body]
				else:
					avg_total[model]["dir_avg"][dir][0] += score_face
					avg_total[model]["dir_avg"][dir][1] += score_body
		avg_total[model]["face"] /= dir_len * label_size
		avg_total[model]["body"] /= dir_len * label_size
		for label in avg_total[model]["label_avg"].keys():
				avg_total[model]["label_avg"][label][0] /= dir_len
				avg_total[model]["label_avg"][label][1] /= dir_len	
		for dir in avg_total[model]["dir_avg"].keys():
				avg_total[model]["dir_avg"][dir][0] /= label_size
				avg_total[model]["dir_avg"][dir][1] /= label_size
		face_ratio = []
		body_ratio = []
		face_ratio.append(round(avg_total[model]['face'].item(),4))
		body_ratio.append(round(avg_total[model]['body'].item(),4))
		for label in avg_total[model]['label_avg'].keys():
			face_ratio.append(round(avg_total[model]['label_avg'][label][0].item(),4))	
			body_ratio.append(round(avg_total[model]['label_avg'][label][1].item(),4))
		for dir in avg_total[model]['dir_avg'].keys():
			face_ratio.append(round(avg_total[model]['dir_avg'][dir][0].item(),4))	
			body_ratio.append(round(avg_total[model]['dir_avg'][dir][1].item(),4))
		
		df = pd.DataFrame({'Face %': face_ratio, 'Body %': body_ratio})
		df.index = ['overall'] + list(avg_total[model]['label_avg'].keys()) + list(avg_total[model]['dir_avg'].keys())

		df["total"] = df["Face %"] + df["Body %"]
		df = df.sort_values(by="total", ascending=False)
		df.to_csv(f"activation_ratio/activation_ratio_{model}.csv")
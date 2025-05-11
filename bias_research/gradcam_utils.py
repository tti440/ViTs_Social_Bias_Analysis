
import copy
import math
from functools import partial

import mmcv
import numpy as np
import torch.nn as nn
from mmcv.transforms import Compose
from mmengine.dataset import default_collate
from mmengine.utils import to_2tuple
from mmengine.utils.dl_utils import is_norm
from mmpretrain.registry import TRANSFORMS

try:
	import pytorch_grad_cam as cam
	from pytorch_grad_cam.activations_and_gradients import \
		ActivationsAndGradients
	from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
	raise ImportError('Please run `pip install "grad-cam>=1.3.6"` to install '
					  '3rd party package pytorch_grad_cam.')
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad
from pytorch_grad_cam.utils.image import preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit

# Alias name
METHOD_MAP = {
	'gradcam++': cam.GradCAMPlusPlus,
}
METHOD_MAP.update({
	cam_class.__name__.lower(): cam_class
	for cam_class in cam.base_cam.BaseCAM.__subclasses__()
})

def reshape_transform(tensor, model):
	"""Build reshape_transform for `cam.activations_and_grads`, which is
	necessary for ViT-like networks."""
	# ViT_based_Transformers have an additional clstoken in features
	if tensor.ndim == 4:
		# For (B, C, H, W)
		return tensor
	elif tensor.ndim == 3:
		num_extra_tokens = getattr(
			model.backbone, 'num_extra_tokens', 1)

		tensor = tensor[:, num_extra_tokens:, :]
		# get heat_map_height and heat_map_width, preset input is a square
		heat_map_area = tensor.size()[1]
		height, width = to_2tuple(int(math.sqrt(heat_map_area)))
		assert height * height == heat_map_area, \
			(f"The input feature's length ({heat_map_area+num_extra_tokens}) "
			 f'minus num-extra-tokens ({num_extra_tokens}) is {heat_map_area},'
			 ' which is not a perfect square number. Please check if you used '
			 'a wrong num-extra-tokens.')
		# (B, L, C) -> (B, H, W, C)
		result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
		# (B, H, W, C) -> (B, C, H, W)
		result = result.permute(0, 3, 1, 2)
		return result
	else:
		raise ValueError(f'Unsupported tensor shape {tensor.shape}.')


def init_cam(method, model, target_layers, use_cuda, reshape_transform):
	"""Construct the CAM object once, In order to be compatible with
	mmpretrain, here we modify the ActivationsAndGradients object."""
	GradCAM_Class = METHOD_MAP[method.lower()]
	cam = GradCAM_Class(
		model=model, target_layers=target_layers)
	# Release the original hooks in ActivationsAndGradients to use
	# ActivationsAndGradients.
	cam.activations_and_grads.release()
	cam.activations_and_grads = ActivationsAndGradients(
		cam.model, cam.target_layers, reshape_transform)

	return cam


def get_layer(layer_str, model):
	"""get model layer from given str."""
	for name, layer in model.named_modules():
		if name == layer_str:
			return layer
	raise AttributeError(
		f'Cannot get the layer "{layer_str}". Please choose from: \n' +
		'\n'.join(name for name, _ in model.named_modules()))


def show_cam_grad(grayscale_cam, src_img, title, out_path=None):
	"""fuse src_img and grayscale_cam and show or save."""
	grayscale_cam = grayscale_cam[0, :]
	src_img = np.float32(src_img) / 255
	visualization_img = show_cam_on_image(
		src_img, grayscale_cam, use_rgb=False)

	if out_path:
		mmcv.imwrite(visualization_img, str(out_path))
	else:
		mmcv.imshow(visualization_img, win_name=title)
	return visualization_img

def get_default_target_layers(model):
	"""get default target layers from given model, here choose nrom type layer
	as default target layer."""
	norm_layers = [
		(name, layer)
		for name, layer in model.backbone.named_modules(prefix='backbone')
		if is_norm(layer)
	]
	if True:
		# For ViT models, the final classification is done on the class token.
		# And the patch tokens and class tokens won't interact each other after
		# the final attention layer. Therefore, we need to choose the norm
		# layer before the last attention layer.
		num_extra_tokens = getattr(
			model.backbone, 'num_extra_tokens', 1)

		# models like swin have no attr 'out_type', set out_type to avg_featmap
		out_type = getattr(model.backbone, 'out_type', 'avg_featmap')
		if out_type == 'cls_token' or num_extra_tokens > 0:
			# Assume the backbone feature is class token.
			name, layer = norm_layers[-3]
			print('Automatically choose the last norm layer before the '
				  f'final attention block "{name}" as the target layer.')
			return [layer]

	# For CNN models, use the last norm layer as the target-layer
	name, layer = norm_layers[-1]
	print('Automatically choose the last norm layer '
		  f'"{name}" as the target layer.')
	return [layer]

from mmpretrain import ImageClassificationInferencer
def return_img(model, cfg, img, method, targets, layer=None, beit=False, milan=False):
	# build the model from a config file and a checkpoint file
	# apply transform and perpare data
	
	transforms = Compose(
		[TRANSFORMS.build(t) for t in cfg.test_dataloader.dataset.pipeline])
	data = transforms({'img_path': img})
	src_img = copy.deepcopy(data['inputs']).numpy().transpose(1, 2, 0)
	data = model.data_preprocessor(default_collate([data]), False)

	# build target layers
	if layer:
		target_layers = [model.backbone.layers[layer].ln1]
	elif beit:
		target_layers = [model.backbone.layers[-1]]
	else:
		try:
			target_layers = [model.backbone.layers[-1].ln1]
		except:
			target_layers = [model.backbone.stages[-1].blocks[-1].norm1]
			
	# For milan
	if milan:
		for param in model.backbone.parameters():
			param.requires_grad = True

	cam = init_cam(method, model, target_layers, 'cpu',
				   partial(reshape_transform, model=model))

	# calculate cam grads and show|save the visualization image
	grayscale_cam = cam(
		data['inputs'],
		targets,
		eigen_smooth=True,
		aug_smooth=True,
	)
	visualization_img=show_cam_grad(
		grayscale_cam, src_img, title=method, out_path="/content/result.jpg")

	return visualization_img, grayscale_cam, cam.activations_and_grads.activations, cam.activations_and_grads.gradients

def generate_gradcam(model, img_path, method="gradcam", targets=None):
	"""
	Generate Grad-CAM visualization for the given model and image.

	Args:
		model: PyTorch model (e.g., DINO Vision Transformer).
		img_path: Path to the input image.
		method: Grad-CAM method to use. Options: "gradcam", "gradcam++", "xgradcam", etc.
		targets: List of ClassifierOutputTarget for specific target classes. If None, it uses the highest scoring category.

	Returns:
		cam_image: Image with Grad-CAM heatmap overlaid on the original image.
	"""
	methods = {
		"gradcam": GradCAM,
		"scorecam": ScoreCAM,
		"gradcam++": GradCAMPlusPlus,
		"ablationcam": AblationCAM,
		"xgradcam": XGradCAM,
		"eigencam": EigenCAM,
		"eigengradcam": EigenGradCAM,
		"layercam": LayerCAM,
		"fullgrad": FullGrad,
	}

	if method not in methods:
		raise ValueError(f"Invalid method '{method}'. Choose from: {list(methods.keys())}")

	# Define target layers (e.g., last LayerNorm in DINO)
	if hasattr(model.backbone, "layers"):
		target_layers = [model.backbone.layers[-1].ln1]
	else:
		target_layers = [model.backbone.blocks[-1].norm1]

	# Initialize the CAM method
	if method == "ablationcam":
		cam = methods[method](model=model, target_layers=target_layers, reshape_transform=reshape_transform1, ablation_layer=AblationLayerVit())
	else:
		cam = methods[method](model=model, target_layers=target_layers, reshape_transform=reshape_transform1)

	# Load and preprocess the input image
	rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]  # Convert BGR to RGB
	rgb_img = cv2.resize(rgb_img, (224, 224))
	rgb_img = np.float32(rgb_img) / 255.0  # Normalize to [0, 1]
	input_tensor = preprocess_image(
		rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
	)

	# Generate the Grad-CAM heatmap
	grayscale_cam = cam(input_tensor=input_tensor, targets=targets, eigen_smooth=True, aug_smooth=True)
	grayscale_cam = grayscale_cam[0]  # Take the first (and only) image in the batch

	# Overlay Grad-CAM on the image
	cam_image = show_cam_on_image(rgb_img, grayscale_cam)

	return cam_image, grayscale_cam, cam.activations_and_grads.activations, cam.activations_and_grads.gradients


def reshape_transform1(tensor):
	"""
	Reshape the tensor for Vision Transformers.
	Args:
		tensor: Input tensor from ViT [batch_size, tokens, embedding_dim].
	Returns:
		Transformed tensor with shape [batch_size, channels, height, width].
	"""
	# Compute spatial dimensions dynamically
	num_tokens = tensor.size(1) - 1  # Exclude class token
	height = width = int(num_tokens ** 0.5)
	result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
	result = result.permute(0, 3, 1, 2)  # Bring channels to the first dimension
	return result


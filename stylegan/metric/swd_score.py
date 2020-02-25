# https://github.com/koshian2/swd-pytorch/blob/master/swd.py
from PIL import Image
import math
import numpy as np
import torch
import torch.nn.functional as F
import torchvision


# Gaussian blur kernel
def get_gaussian_kernel(device="cpu"):
	kernel = np.array([
		[1, 4, 6, 4, 1],
		[4, 16, 24, 16, 4],
		[6, 24, 36, 24, 6],
		[4, 16, 24, 16, 4],
		[1, 4, 6, 4, 1]], np.float32) / 256.0
	gaussian_k = torch.as_tensor(kernel.reshape(1, 1, 5, 5)).to(device)
	return gaussian_k


def pyramid_down(image, device="cpu"):
	gaussian_k = get_gaussian_kernel(device=device)
	# channel-wise conv(important)
	multiband = [F.conv2d(image[:, i:i + 1, :, :], gaussian_k, padding=2, stride=2) for i in range(3)]
	down_image = torch.cat(multiband, dim=1)
	return down_image


def pyramid_up(image, device="cpu"):
	gaussian_k = get_gaussian_kernel(device=device)
	upsample = F.interpolate(image, scale_factor=2)
	multiband = [F.conv2d(upsample[:, i:i + 1, :, :], gaussian_k, padding=2) for i in range(3)]
	up_image = torch.cat(multiband, dim=1)
	return up_image


def gaussian_pyramid(original, n_pyramids, device="cpu"):
	x = original
	# pyramid down
	pyramids = [original]
	for i in range(n_pyramids):
		x = pyramid_down(x, device=device)
		pyramids.append(x)
	return pyramids


def laplacian_pyramid(original, n_pyramids, device="cpu"):
	# create gaussian pyramid
	pyramids = gaussian_pyramid(original, n_pyramids, device=device)

	# pyramid up - diff
	laplacian = []
	for i in range(len(pyramids) - 1):
		diff = pyramids[i] - pyramid_up(pyramids[i + 1], device=device)
		laplacian.append(diff)
	# Add last gaussian pyramid
	laplacian.append(pyramids[len(pyramids) - 1])
	return laplacian


def minibatch_laplacian_pyramid(image, n_pyramids, batch_size, device="cpu"):
	n = image.size(0) // batch_size + np.sign(image.size(0) % batch_size)
	pyramids = []
	for i in range(n):
		x = image[i * batch_size:(i + 1) * batch_size]
		p = laplacian_pyramid(x.to(device), n_pyramids, device=device)
		p = [x.cpu() for x in p]
		pyramids.append(p)
	del x
	result = []
	for i in range(n_pyramids + 1):
		x = []
		for j in range(n):
			x.append(pyramids[j][i])
		result.append(torch.cat(x, dim=0))
	return result


def extract_patches(pyramid_layer, slice_indices,
                    slice_size=7, unfold_batch_size=128, device="cpu"):
	assert pyramid_layer.ndim == 4
	n = pyramid_layer.size(0) // unfold_batch_size + np.sign(pyramid_layer.size(0) % unfold_batch_size)
	# random slice 7x7
	p_slice = []
	for i in range(n):
		# [unfold_batch_size, ch, n_slices, slice_size, slice_size]
		ind_start = i * unfold_batch_size
		ind_end = min((i + 1) * unfold_batch_size, pyramid_layer.size(0))
		x = pyramid_layer[ind_start:ind_end].unfold(
			2, slice_size, 1).unfold(3, slice_size, 1).reshape(
			ind_end - ind_start, pyramid_layer.size(1), -1, slice_size, slice_size)
		# [unfold_batch_size, ch, n_descriptors, slice_size, slice_size]
		x = x[:, :, slice_indices, :, :]
		# [unfold_batch_size, n_descriptors, ch, slice_size, slice_size]
		p_slice.append(x.permute([0, 2, 1, 3, 4]))
	# sliced tensor per layer [batch, n_descriptors, ch, slice_size, slice_size]
	x = torch.cat(p_slice, dim=0)
	# normalize along ch
	std, mean = torch.std_mean(x, dim=(0, 1, 3, 4), keepdim=True)
	x = (x - mean) / (std + 1e-8)
	# reshape to 2rank
	x = x.reshape(-1, 3 * slice_size * slice_size)
	return x


def calculate_swd(image1, image2,
                  n_pyramids=None, slice_size=7, n_descriptors=128,
                  n_repeat_projection=128, proj_per_repeat=4, device="cpu", return_by_resolution=False,
                  pyramid_batchsize=128, enforce_balance=True):
	# n_repeat_projectton * proj_per_repeat = 512
	# Please change these values according to memory usage.
	# original = n_repeat_projection=4, proj_per_repeat=128
	assert image1.ndim == 4 and image2.ndim == 4
	if enforce_balance:
		assert image1.size() == image2.size()

	if n_pyramids is None:
		n_pyramids = int(np.rint(np.log2(image1.size(2) // 16)))
	with torch.no_grad():
		# minibatch laplacian pyramid for cuda memory reasons
		pyramid1 = minibatch_laplacian_pyramid(image1, n_pyramids, pyramid_batchsize, device=device)
		pyramid2 = minibatch_laplacian_pyramid(image2, n_pyramids, pyramid_batchsize, device=device)
		result = []

		for i_pyramid in range(n_pyramids + 1):
			# indices
			n = (pyramid1[i_pyramid].size(2) - 6) * (pyramid1[i_pyramid].size(3) - 6)
			indices = torch.randperm(n)[:n_descriptors]

			# extract patches on CPU
			# patch : 2rank (n_image*n_descriptors, slice_size**2*ch)
			p1 = extract_patches(pyramid1[i_pyramid], indices,
			                     slice_size=slice_size, device="cpu")
			p2 = extract_patches(pyramid2[i_pyramid], indices,
			                     slice_size=slice_size, device="cpu")

			p1, p2 = p1.to(device), p2.to(device)

			distances = []
			for j in range(n_repeat_projection):
				# random
				rand = torch.randn(p1.size(1), proj_per_repeat).to(device)  # (slice_size**2*ch)
				rand = rand / torch.std(rand, dim=0, keepdim=True)  # noramlize
				# projection
				proj1 = torch.matmul(p1, rand)
				proj2 = torch.matmul(p2, rand)

				if not enforce_balance and proj1.size() != proj2.size():
					max_size = max(proj1.size(0), proj2.size(0))
					if proj1.size(0) < max_size:
						proj1 = proj1.repeat(math.ceil(max_size / proj1.size(0)), 1)[:max_size]
					else:
						proj2 = proj2.repeat(math.ceil(max_size / proj2.size(0)), 1)[:max_size]

				proj1, _ = torch.sort(proj1, dim=0)
				proj2, _ = torch.sort(proj2, dim=0)
				d = torch.abs(proj1 - proj2)
				distances.append(torch.mean(d))

			# swd
			result.append(torch.mean(torch.stack(distances)))

		# average over resolution
		result = torch.stack(result) * 1e3
		if return_by_resolution:
			return result.cpu()
		else:
			return torch.mean(result).cpu()
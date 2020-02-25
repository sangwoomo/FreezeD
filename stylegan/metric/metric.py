import time
import functools
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset, DataLoader

from .fid_score import calculate_frechet_distance
# from .kid_score import polynomial_mmd_averages
from .swd_score import calculate_swd


def get_fake_images_and_acts(inception, g_running, code_size, step, alpha, sample_num=5000, batch_size=16):
	dataset = TensorDataset(torch.randn(sample_num, code_size))
	loader = DataLoader(dataset, batch_size=batch_size * torch.cuda.device_count())

	pbar = tqdm(total=sample_num, position=1, leave=False)
	pbar.set_description('Get fake images and acts')

	images = []
	acts = []
	for gen_in in loader:
		gen_in = gen_in[0].cuda()  # list -> tensor
		with torch.no_grad():
			fake_image = g_running(gen_in, step=step, alpha=alpha)
			out = inception(fake_image)
			out = out[0].squeeze(-1).squeeze(-1)

		images.append(fake_image.cpu())  # cuda tensor
		acts.append(out.cpu().numpy())  # numpy

		pbar.update(len(gen_in))

	images = torch.cat(images, dim=0)  # N x C x H x W
	acts = np.concatenate(acts, axis=0)  # N x d

	return images, acts


def compute_time(func):
	@functools.wraps(func)
	def wrapper(*args, **kwargs):
		begin = time.time()
		result = func(*args, **kwargs)
		end = time.time()
		print(f'function: {func}\tcomputation time: {round(end - begin)}s')
		return result
	return wrapper


# @compute_time
def compute_fid(real_acts, fake_acts):
	mu1, sigma1 = (np.mean(real_acts, axis=0), np.cov(real_acts, rowvar=False))
	mu2, sigma2 = (np.mean(fake_acts, axis=0), np.cov(fake_acts, rowvar=False))

	fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

	return fid


# @compute_time
def compute_kid(real_acts, fake_acts):
	# size = min(len(real_acts), len(fake_acts))
	# mmds, mmd_vars = polynomial_mmd_averages(real_acts[:size], fake_acts[:size])
	mmds, mmd_vars = polynomial_mmd_averages(real_acts, fake_acts, replace=True)
	kid = mmds.mean()

	# print("mean MMD^2 estimate:", mmds.mean())
	# print("std MMD^2 estimate:", mmds.std())
	# print("MMD^2 estimates:", mmds, sep='\n')
	#
	# print("mean Var[MMD^2] estimate:", mmd_vars.mean())
	# print("std Var[MMD^2] estimate:", mmd_vars.std())
	# print("Var[MMD^2] estimates:", mmd_vars, sep='\n')

	return kid


# @compute_time
def compute_swd(real_images, fake_images):
	# size = min(len(real_images), len(fake_images))
	# swd = calculate_swd(real_images[:size], fake_images[:size], device="cuda")
	swd = calculate_swd(real_images, fake_images, device="cuda", enforce_balance=False)

	return swd



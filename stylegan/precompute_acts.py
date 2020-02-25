import argparse
import pickle
import random
import numpy as np
from tqdm import tqdm

import torch
from torchvision import transforms

from dataset import MultiResolutionDataset
from train import sample_data
from metric.inception import InceptionV3


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Pre-compute activations')

	parser.add_argument('--dataset', type=str, required=True, help='dataset name')
	parser.add_argument('--seed', type=int, default=0, help='random seed')
	parser.add_argument('--image_size', type=int, default=256, help='image size')
	parser.add_argument('--batch_size', type=int, default=50, help='batch size')

	args = parser.parse_args()

	random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	inception = InceptionV3().cuda()

	transform = transforms.Compose([
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
	])

	dataset = MultiResolutionDataset(f'./dataset/{args.dataset}_lmdb', transform)
	loader = sample_data(dataset, args.batch_size, args.image_size)

	pbar = tqdm(total=len(dataset))

	acts = []
	for real_index, real_image in loader:
		real_image = real_image.cuda()
		with torch.no_grad():
			out = inception(real_image)
			out = out[0].squeeze(-1).squeeze(-1)
		acts.append(out.cpu().numpy())
		pbar.update(len(real_image))
	acts = np.concatenate(acts, axis=0)

	with open(f'dataset/{args.dataset}_acts.pickle', 'wb') as handle:
		pickle.dump(acts, handle)


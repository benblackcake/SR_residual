import cv2
import os 
import glob
import numpy as np
import matplotlib.pyplot as plt
import random
import h5py

import argparse


def input_set(args):
	train_filenames = np.array(glob.glob(os.path.join(args.train_dir, '**', '*.*'), recursive=True))
	val_filenames = np.array(glob.glob(os.path.join('Benchmarks', '**', '*_HR.png'), recursive=True))

	return train_filenames

def get_batchs(datas, batch_size, scale):
	num_samples = len(datas)
	batches = range(0, num_samples - batch_size + 1, batch_size)

	for batch in batches:
		li_lr = []
		li_hr = []
		X_batch = datas[batch:batch + batch_size]
		for data in X_batch:
			img = cv2.imread(data)
			lr_img = cv2.resize(img,None,fx = 1.0/scale ,fy=1.0/scale, interpolation = cv2.INTER_CUBIC)
			lr_img = cv2.resize(img,None,fx = scale ,fy=scale, interpolation = cv2.INTER_CUBIC)

			li_hr.append(img)
			li_lr.append(lr_img)

		li_lr = np.asarray(li_lr)
		li_hr = np.asarray(li_hr)
		yield li_lr, li_hr



# if __name__=='__main__':
# 	parser = argparse.ArgumentParser()
# 	parser.add_argument('--train-dir', type=str, default='Train', help='Directory containing training images')
# 	parser.add_argument('--is-val', action='store_true')
# 	parser.add_argument('--is-eval', action='store_true')
# 	parser.add_argument('--val-dir', type=str, default='Benchmarks/', help='Directory containg imagrs for validation')
# 	parser.add_argument('--batch-size', type=int, default=16, help='Mini-batch size.')
# 	parser.add_argument('--img-size', type=int, default=96, help='Mini-batch size.')
# 	args = parser.parse_args()


# 	data = input_setup(args)

# 	for i,j  in get_batchs(data, 16,2):
# 		print(i.shape)
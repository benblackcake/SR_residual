
import tensorflow as tf
import numpy as np
import argparse
import sys

from model import SRresidual
from utils import *
from tqdm import tqdm,trange



def load(sess, saver, checkpoint_dir):
    """
        To load the checkpoint use to test or pretrain
    """
    print("\nReading Checkpoints.....\n\n")
    model_dir = "%s_%s" % ("SR_residual", 33)# give the model name by label_size
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    print(checkpoint_dir)
    # Check the checkpoint is exist 
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_path = str(ckpt.model_checkpoint_path) # convert the unicode to string
        saver.restore(sess, os.path.join(os.getcwd(), ckpt_path))
        print("\n Checkpoint Loading Success! %s\n\n"% ckpt_path)
    else:
        print("\n! Checkpoint Loading Failed \n\n")

def save(sess, saver, checkpoint_dir, step):
    """
        To save the checkpoint use to test or pretrain
    """
    model_name = "SR_residual.model"
    model_dir = "%s_%s" % ("SR_residual", 33)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
         os.makedirs(checkpoint_dir)

    saver.save(sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)


def main():
    
	parser = argparse.ArgumentParser()
	parser.add_argument('--learning-rate', type=float, default=1e-5, help='Learning rate for Adam.')
	parser.add_argument('--epoch', type=int, default='10000', help='How many iterations ')
	parser.add_argument('--image-size', type=int, default=33, help='Size of random crops used for training samples.')
	parser.add_argument('--c-dim', type=int, default=3, help='The size of channel')
	parser.add_argument('--scale', type=int, default=2, help='the size of scale factor for preprocessing input image')
	parser.add_argument('--checkpoint-dir', type=str, default='checkpoint', help='Name of checkpoint directory')
	parser.add_argument('--result_dir', type=str, default='result', help='Name of result directory')
	parser.add_argument('--test-img', type=str, default='', help='test_img')
	parser.add_argument('--is-train', type=int, default=1, help='training')
	parser.add_argument('--batch-size', type=int, default=64, help='Mini-batch size.')

	args = parser.parse_args()


	input_x = tf.placeholder(tf.float32, [None, None, None, 1], name='input_lowres')
	input_y = tf.placeholder(tf.float32, [None, None, None, 1], name='input_highres')

	sr_residual = SRresidual(args.learning_rate)
	predict_residual = sr_residual.forward(input_x, 20)
	r = tf.abs(input_y-input_x)
	loss_func = sr_residual.loss(r, predict_residual)

	optimizer = sr_residual.optmizer(loss_func)

	train_data_path = 'done_dataset\PreprocessedData.h5'

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		# nx, ny = input_setup(args.image_size, args.scale, args.is_train, args.checkpoint_dir)
		# data_dir = checkpoint_dir(args.is_train, args.checkpoint_dir)
		# input_, label_ = read_data(data_dir)
		train_data_set = get_data_set(train_data_path,'train')
	
		saver = tf.train.Saver()
		counter = 0
		load(sess, saver, args.checkpoint_dir)

		if args.is_train:
			# pbar = tqdm(range(args.epoch),bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
			for epoch in range(args.epoch):
				# batch_idxs = len(input_) // args.batch_size
				t =tqdm(range(0, len(train_data_set) - args.batch_size + 1, args.batch_size),bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

		        # print(len(input_))
				for batch_idx in t:
					batch_hr = train_data_set[batch_idx:batch_idx + args.batch_size]
					batch_lr = downsample_batch(batch_hr, factor=args.scale)
					batch_lr, batch_hr = pre_process(batch_lr, batch_hr)
					
					b_images = np.reshape(batch_lr[:,:,:,0],[args.batch_size,args.image_size,args.image_size,1])
					b_labels = np.reshape(batch_hr[:,:,:,0],[args.batch_size,args.image_size,args.image_size,1])
					_, err = sess.run([optimizer, loss_func],
								feed_dict={input_x: b_images, input_y: b_labels})

					# debug_shape = sess.run([lr_images],feed_dict={lr_images: b_images, hr_images: b_labels})
					# debug_shape = np.asarray(debug_shape)
					# print(debug_shape.shape)

					counter +=1
					# print('error: ',err)
					t.set_description('[Epoch: %s][Step: %s][loss: %.8f]' %(epoch, counter, err))
					if counter % 500 == 0:
					    save(sess, saver, args.checkpoint_dir, counter)

		else:
			print("Now Start Testing...")
			# image = imread('Test/Set5/butterfly_GT.bmp')
			# image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)

			input_LR, input_HR = preprocess('Test/Set5/butterfly_GT.bmp', scale = args.scale)

			input_LR = input_LR/255.
			input_HR = input_HR/255.

			input_image = input_LR[:,:,0]
			input_image = np.reshape(input_image,[1,input_image.shape[0],input_image.shape[1],1])
			# input_image = input_image[:,:,:,np.newaxis]

			result = predict_residual.eval({input_x: input_image})
			print(result.shape)
			result = np.squeeze(result)
			# result = merge(result,[nx,ny])

			# result = np.ceil(result)
			print(result)
			# lr_image = merge(input_,[nx,ny], c_dim=3)
			checkimage('bicubic_debug.bmp', input_LR*255)
			# label_iamge = merge(label_,[nx,ny], c_dim=3) *255
			sr_image = input_LR

			print(sr_image[:,:,0])
			print(input_LR[:,:,0])

			sr_image[:,:,0] =np.abs(sr_image[:,:,0]+ result)
			# sr_image = np.ceil(sr_image)
			print(sr_image[:,:,0])
			print(input_LR[:,:,0])

			# result = cv2.cvtColor(result,cv2.COLOR_YCrCb2RGB)
			cv2.imwrite('residual_debug.bmp',result*255)     

			checkimage('label_debug.bmp', input_HR*255)
			checkimage('sr_result_debug.bmp', sr_image*255)

			plt.imshow(result, cmap='gray')
			plt.show()
			print('__debug__result_testing...')
			print(result.shape)

if __name__ =='__main__':
	main()


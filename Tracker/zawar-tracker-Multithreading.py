import logging
import time
import tensorflow as tf
import os
import goturn_net
import cv2
import os
import numpy as np
from stackclass import Stack
import scipy.misc
from multiprocessing import Queue, Pool
import multiprocessing


NUM_EPOCHS = 500
BATCH_SIZE = 10
WIDTH = 227
HEIGHT = 227

x1 = 0.0  
y1 = 0.0
x2 = 0.0
y2 = 0.0

previous = cv2.imread("Path of Your Initial Image tracker need as search Image")

tracknet = goturn_net.TRACKNET(BATCH_SIZE, train = True)
tracknet.build()

def next_batch(tensors_search_, tensors_target_, box_tensors_ , min_queue_examples=128,num_threads=8): 		
	[search_batch, target_batch, box_batch] = tf.train.batch(
		[tensors_search_, tensors_target_, box_tensors_],
		batch_size=BATCH_SIZE,
		num_threads=num_threads,
		capacity=min_queue_examples + (num_threads+2)*BATCH_SIZE)

	return [search_batch[:,0,:,:,:], target_batch[:,0,:,:,:], box_batch[:,0,:]]

def detect_object(latest, sess):
	global x1, y1, x2, y2, previous
	print "Before update"
	print "X1:", x1, "Y1:", y1, "X2:", x2, "Y2:", y2 	
	intial_box = [float(x1),float(y1),float(x2),float(y2)]
	ttenslist = []
	stenslist = []
	min_queue_examples=128
	num_threads=1	
	search_batch=0
	target_batch=0
	box_batch=0     
				
	#data reader
	
	search_tensor = tf.image.resize_images(previous,[HEIGHT,WIDTH],method=tf.image.ResizeMethod.BILINEAR)
	search_tensor = tf.reshape(search_tensor , shape=[1,HEIGHT,WIDTH,3])
	stenslist.append(search_tensor)
	tensors_search_ = tf.concat(stenslist , axis=0)
	print "Data reader Target Tensor"
	target_tensor = tf.image.resize_images(latest,[HEIGHT,WIDTH],method=tf.image.ResizeMethod.BILINEAR)    
	target_tensor = tf.reshape(target_tensor , shape=[1,HEIGHT,WIDTH,3])
	ttenslist.append(target_tensor)
	tensors_target_ = tf.concat(ttenslist , axis=0)
	print "Data reader Box Tensor"
	box_tensor = []
	box_tensor.append(intial_box)
	box_tensor = np.array(box_tensor)
	box_tensors_ = tf.convert_to_tensor(box_tensor, dtype=tf.float64)

	print "In next batch part"
	#next batch
	batch_queue = next_batch(tensors_search_, tensors_target_, box_tensors_)
	
	print "Creating Session"
	
	coord = tf.train.Coordinator()
	tf.train.start_queue_runners(sess=sess, coord=coord)		
 
	print "Into Current Batch"	    	
	cur_batch = sess.run(batch_queue)

	print "get FC4"	
	[batch_loss, fc4] = sess.run([tracknet.loss, tracknet.fc4],feed_dict={tracknet.image:cur_batch[0],
			tracknet.target:cur_batch[1], tracknet.bbox:cur_batch[2]})
	print "Bounding box drawing"
	previous = latest
	bbox=[] 
	innerloop = [x for x in fc4]
	print "innerloop",innerloop	
	bbox = [y for y in innerloop]
	x1 = float(bbox[0][0])
	y1 = float(bbox[0][1])	
	x2 = float(bbox[0][2])
	y2 = float(bbox[0][3])
	print "After update"
	print "X1:", x1, "Y1:", y1, "X2:", x2, "Y2:", y2
	ttl=(int(float(x1)*50), int(float(y1)*50))	
	tbr=(int( float(x2)*50 + float(x1) ), int( float(y2)*50 + float(y1) ))
	#p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
	cv2.rectangle(latest, ttl, tbr, (0,255,0), 2)
	#cv2.rectangle(latest, (xx1,yy1),(xx1+227, yy1+227),(0, 255, 0), 2)
	print "frame with bounding box"
	return latest		


def worker(input_q, output_q):
	global previous	
	global previous, latest
	sess = tf.Session()	
	init = tf.global_variables_initializer()
	init_local = tf.local_variables_initializer()
	sess.run(init)
	sess.run(init_local)
	
	print "Loading Model"		
	ckpt_dir = "./checkpoints"
	if not os.path.exists(ckpt_dir):
		os.makedirs(ckpt_dir)
	ckpt = tf.train.get_checkpoint_state(ckpt_dir)
	if ckpt and ckpt.model_checkpoint_path:
		saver = tf.train.Saver()
		saver.restore(sess, ckpt.model_checkpoint_path)

	while True:
		#if input_q.qsize() > 3:
			#previous = input_q.get() 
		latest = input_q.get()
		output_q.put(detect_object(latest, sess))

	fps.stop()
	sess.close()



	
if __name__ == "__main__":
	
	#webcap = cv2.VideoCapture('./video/testing.mp4')
	cap = cv2.VideoCapture(0)
	start = time.time()
	max_Frames = 10
	newfps = 0 
	frame_idx = 0	 
	frame_interval=12
	input_q  = Queue(5)
	output_q = Queue(5)
	pool = Pool(1, worker, (input_q, output_q))
	
	count = 0	
	while True:
	   ret,frame = cap.read()

	   if ret:
	   		count+=1	
	   		if(count%frame_interval == 0):		
			   	frame_idx += 1
			   	input_q.put(frame)
			   	if output_q.empty():
				   	pass  # fill up queue
			   	else:
				   	#cv2.imshow('Video', scipy.misc.imresize(output_q.get(), (560,1200)))
				   	cv2.imshow('Video', scipy.misc.imresize(output_q.get(), (480,600)))
				   	if frame_idx == max_Frames:
					   	end = time.time()
					   	seconds = end - start
					   	newfps  = frame_idx / seconds;
					   	print "Estimated frames per second : {0}".format(newfps);

		   	if cv2.waitKey(1) & 0xFF == ord('q'):
			   	break;
	
	pool.terminate()
	cap.release()
	cv2.destroyAllWindows()

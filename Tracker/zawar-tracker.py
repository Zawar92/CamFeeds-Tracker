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
from multiprocessing import Queue


NUM_EPOCHS = 500
BATCH_SIZE = 10
WIDTH = 227
HEIGHT = 227


def next_batch(tensors_search_, tensors_target_, box_tensors_ , min_queue_examples=128,num_threads=1): 		
    [search_batch, target_batch, box_batch] = tf.train.batch(
        [tensors_search_, tensors_target_, box_tensors_],
        batch_size=BATCH_SIZE,
        num_threads=num_threads,
        capacity=min_queue_examples + (num_threads+2)*BATCH_SIZE)

    return [search_batch[:,0,:,:,:], target_batch[:,0,:,:,:], box_batch[:,0,:]]

def detect_object(previous,latest,tracknet,sess):
    x1 = float(2.43045) 
    y1 = float(2.50264) 
    x2 = float(7.33545) 
    y2 = float(7.51349) 	
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
    bbox=[] 
    innerloop = [x for x in fc4]
    bbox = [y for y in innerloop]
    xx1 = bbox[0][0]
    yy1 = bbox[0][1]	
    xx2 = bbox[0][2]
    yy2 = bbox[0][3]
    print "X1:", xx1
    print "Y1:", yy1
    print "X2:", xx2
    print "Y2:", yy2	   
    ttl=(int(float(xx1)*50), int(float(yy1)*50))	
    tbr=(int(float(xx2)*50), int(float(yy2)*50))			
    cv2.rectangle(latest, ttl, tbr, (255,0,0), 2)
    print "frame with bounding box"		
    return output_q.put(latest)		


	
if __name__ == "__main__":
    
    tracknet = goturn_net.TRACKNET(BATCH_SIZE, train = True)
    tracknet.build()

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
	
    #webcap = cv2.VideoCapture('./video/testing.mp4')
    webcap = cv2.VideoCapture(0)
    start = time.time()
    max_Frames = 10
    newfps = 0 
    frame_idx = 0	    
    input_q  = Queue(5)
    output_q = Queue(5)
    while True:
        ret,frame = webcap.read()
        if ret:
	    input_q.put(frame)
	    if input_q.qsize() > 2:	
                    previous = input_q.get()
		    latest = input_q.get()			
		    frame_idx += 1
		    #cv2.imshow('Video', scipy.misc.imresize(detect_object(previous,latest,tracknet,sess), (480,840)))
		    detect_object(previous,latest,tracknet,sess)
		    cv2.imshow('Video',output_q.get())		
		    #path_output_dir = "./results"
		    #cv2.imwrite(os.path.join(path_output_dir,"%d.png")%frame_idx,img)
		    #frame_idx += 1
		    #print path_output_dir+str(frame_idx)+".png"	
		    if frame_idx == max_Frames:
		        end = time.time()
		        seconds = end - start
		        newfps  = frame_idx / seconds;
		        print("Estimated frames per second : {0}".format(newfps))
		    
		    if cv2.waitKey(1) & 0xFF == ord('q'):
		        break;
	    else:
		pass                         
    webcap.release()
    cv2.destroyAllWindows()

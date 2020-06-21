#coding=utf-8
import tensorflow as tf
import numpy as np
import os
import time
import datetime
from units import load,get_wordDic_Embedding,batch_gen_with_test,batch_gen_with_list_wise,get_overlap_dict
import operator
from model import QINM
import random
import evaluation as evaluation_test
import pickle
from sklearn.model_selection import train_test_split
import configure
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tensorboard_log_dir =  "tensorboard_logs/"
now = int(time.time()) 
timeArray = time.localtime(now)
timeStamp = time.strftime("%Y%m%d%H%M%S", timeArray)
timeDay = time.strftime("%Y%m%d", timeArray)
print (timeStamp)

from functools import wraps

def log_time_delta(func):
    @wraps(func)
    def _deco(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        delta = end - start
        print( "%s runed %.2f seconds"% (func.__name__,delta))
        return ret
    return _deco

FLAGS = configure.flags.FLAGS
FLAGS.flag_values_dict()
# FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print(("{}={}".format(attr.upper(), value)))
log_dir = 'log/'+ timeDay+"/"+FLAGS.data+"/"+FLAGS.file_name
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
data_file = log_dir + '/test_' + FLAGS.data + timeStamp
para_file = log_dir + '/test_' + FLAGS.data + timeStamp + '_para'
precision = data_file + 'precise'


@log_time_delta
def predict(sess,cnn,test,alphabet, batch_size,q_len,d_len):
	scores = []
	d = get_overlap_dict(test,alphabet,q_len,d_len)
	for data in batch_gen_with_test(test,alphabet,batch_size,q_len,d_len,overlap_dict = d):
		feed_dict = {
			cnn.query:data[0],
			cnn.document:data[1],
			cnn.q_overlap:data[2],
			cnn.d_overlap:data[3],
			cnn.tfidf_value:data[4],
			cnn.dropout_keep_prob:1.0
		}
		score = sess.run(cnn.scores,feed_dict)
		
		scores.extend(score)
	return np.array(scores[:len(test)])

@log_time_delta
def train_model():
	train,test = load(FLAGS.data, FLAGS.file_name)
	q_max_sent_length = FLAGS.max_len_query
	d_max_sent_length = FLAGS.max_len_document

	alphabet, embeddings = get_wordDic_Embedding(FLAGS.data, FLAGS.file_name, 50)
	with tf.Graph().as_default():
		with tf.device("/gpu:0"):
			session_conf = tf.ConfigProto()
			session_conf.allow_soft_placement = FLAGS.allow_soft_placement
			session_conf.log_device_placement = FLAGS.log_device_placement
			session_conf.gpu_options.allow_growth = True
		sess = tf.Session(config = session_conf)

		with sess.as_default(),open(precision,"w") as log:
			log.write(str(FLAGS.__flags)+'\n')
			cnn = QINM(
				max_input_query = q_max_sent_length,
				max_input_docu = d_max_sent_length,
				vocab_size = len(alphabet),
				embedding_size = FLAGS.embedding_dim,
				batch_size = FLAGS.batch_size,
				embeddings = embeddings,
				filter_sizes = list(map(int,FLAGS.filter_sizes.split(","))),
				num_filters = FLAGS.num_filters,
				l2_reg_lambda = FLAGS.l2_reg_lambda,
				trainable = FLAGS.trainable,
				overlap_needed = FLAGS.overlap_needed,
				pooling = FLAGS.pooling,
				extend_feature_dim = FLAGS.extend_feature_dim
				)

			cnn.build_graph()
			
			global_step = tf.Variable(0,name = 'global_step',trainable = False)
			learning_rate = FLAGS.learning_rate

			optimizer = tf.train.AdamOptimizer(learning_rate,epsilon=1e-08)
			grads, v = zip(*optimizer.compute_gradients(cnn.loss))
			grads, _ = tf.clip_by_global_norm(grads, 5.0)
			train_op = optimizer.apply_gradients(zip(grads, v), global_step = global_step)

			saver = tf.train.Saver(tf.global_variables(),max_to_keep = 4)		
			
			sess.run(tf.global_variables_initializer())

			map_max = 0.020
			for i in range(FLAGS.num_epochs):
				print ("\nepoch "+str(i)+"\n")
				d = get_overlap_dict(train,alphabet,q_len = q_max_sent_length,d_len = d_max_sent_length)
				datas = batch_gen_with_list_wise(train,alphabet,FLAGS.batch_size,q_len = q_max_sent_length,d_len = d_max_sent_length,overlap_dict = d)				
				j = 1
				for data in datas:
					feed_dict = {
						cnn.query:data[0],
						cnn.document:data[1],
						cnn.input_label:data[2],
						cnn.q_overlap:data[3],
						cnn.d_overlap:data[4],
						cnn.tfidf_value:data[5],
						cnn.dropout_keep_prob:0.5
					}
					_,step,l2_loss,loss= sess.run([train_op,global_step,cnn.l2_loss,cnn.loss],feed_dict)
					print ("{} loss: {}，l2_loss : {}".format(j,loss,l2_loss))
					j+=1
					time_str = datetime.datetime.now().isoformat()

				predicted = predict(sess,cnn,train,alphabet,FLAGS.batch_size,q_max_sent_length,d_max_sent_length)
				
				map_NDCG0_NDCG1_ERR_p_train = evaluation_test.evaluationBypandas(train,predicted[:,-1])
				predicted_test = predict(sess,cnn,test,alphabet,FLAGS.batch_size,q_max_sent_length,d_max_sent_length)				
				
				map_NDCG0_NDCG1_ERR_p_test = evaluation_test.evaluationBypandas(test,predicted_test[:,-1])				

				if map_NDCG0_NDCG1_ERR_p_test[0] > map_max:
					map_max = map_NDCG0_NDCG1_ERR_p_test[0]
					timeStamp = time.strftime("%Y%m%d%H%M%S", time.localtime(int(time.time())))
					folder = 'runs/'+FLAGS.data+"/"+FLAGS.file_name+"/"+ timeDay
					out_dir = folder +'/'+timeStamp+'_'+str(map_NDCG0_NDCG1_ERR_p_test[0])
					if not os.path.exists(folder):
						os.makedirs(folder)
					save_path = saver.save(sess, out_dir)
					print ("Model saved in file: ", save_path)

				print ("{}:train epoch:map,NDCG0,NDCG1,ERR,p {}".format(i,map_NDCG0_NDCG1_ERR_p_train))
				print ("{}:test epoch:map,NDCG0,NDCG1,ERR,p {}".format(i,map_NDCG0_NDCG1_ERR_p_test))
				
				line1 = " {}:epoch: map_train{}".format(i,map_NDCG0_NDCG1_ERR_p_train)
				log.write(line1+"\n")
				line = " {}:epoch: map_test{}".format(i,map_NDCG0_NDCG1_ERR_p_test)
				log.write(line+"\n")
				log.write("\n")
				log.flush()
			log.close()

if __name__ == "__main__":
	if FLAGS.loss == "list_wise":
		train_model()
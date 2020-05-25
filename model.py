#encoding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf 
import numpy as np 
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
rng = np.random.RandomState(23455)


class IR_quantum(object):
	def __init__(
		self, max_input_query,max_input_docu, vocab_size, embedding_size ,batch_size,
		embeddings,filter_sizes,num_filters,l2_reg_lambda = 0.0,trainable = True,
		pooling = 'max',overlap_needed = True,extend_feature_dim = 10):

		# self.dropout_keep_prob = dropout_keep_prob
		self.num_filters = num_filters
		self.embeddings = embeddings
		self.embedding_size = embedding_size
		self.vocab_size = vocab_size
		self.trainable = trainable
		self.filter_sizes = filter_sizes
		self.pooling = pooling
		self.total_embedding_dim = embedding_size
		self.batch_size = batch_size
		self.l2_reg_lambda = l2_reg_lambda
		self.para = []
		self.max_input_query = max_input_query
		self.max_input_docu = max_input_docu
		self.hidden_num = 128
		self.rng = 23455
		self.overlap_need = overlap_needed
		self.extend_feature_dim = extend_feature_dim
		self.conv1_kernel_num = 32
		self.kernel_sizes = [3,4,5]
		self.stdv = 0.5
		

	def creat_placeholder(self):
		self.query = tf.placeholder(tf.int32,[self.batch_size,self.max_input_query],name = "input_query")
		self.document = tf.placeholder(tf.int32,[self.batch_size,self.max_input_docu],name = "input_document")
		self.input_label = tf.placeholder(tf.float32,[self.batch_size,1],name = "input_label")

		self.q_overlap = tf.placeholder(tf.int32,[self.batch_size,self.max_input_query],name = "q_overlap")
		self.d_overlap = tf.placeholder(tf.int32,[self.batch_size,self.max_input_docu],name = "d_overlap")
		self.tfidf_value = tf.placeholder(tf.float32,[self.batch_size,self.max_input_docu],name = 'tfidf_value')
		self.dropout_keep_prob = tf.placeholder(tf.float32,name ="dropout_keep_prob")


	def load_embeddings(self):
		with tf.name_scope("embedding"):
			print ("load embeddings")
			self.words_embeddings = tf.Variable(np.array(self.embeddings),name = "W",dtype = "float32", trainable=True)
			self.overlap_w = tf.get_variable("overlap_w",shape = [3,self.embedding_size],initializer = tf.random_normal_initializer())
			self.query_gate_weight = tf.Variable(tf.random_uniform([self.max_input_query,1],minval=-1*self.stdv, maxval=self.stdv, dtype=tf.float32),trainable = True, name = "char_embedding")
		
		self.embedded_chars_q = self.concat_embedding(self.query,self.q_overlap)
		self.embedded_chars_d = self.concat_embedding(self.document,self.d_overlap)
		
	def concat_embedding(self, words_indice, overlap_indice):
		embedded_chars = tf.nn.embedding_lookup(self.words_embeddings,words_indice)
		overlap_embedding = tf.nn.embedding_lookup(self.overlap_w,overlap_indice)
		if self.overlap_need:
			return tf.reduce_sum([embedded_chars,overlap_embedding],0)
			# return tf.concat([embedded_chars,overlap_embedding],2)
		else:
			return embedded_chars


	def creat_index_matrix(self,batch,word_num):
		all_document = []
		document_index_list = []
		for i in range(word_num):
			document_index_list = []
			for j in range(batch):
				document_i_list = []
				index = []
				index.append(j)
				for k in range(word_num):
					index.append(i)
					document_i_list.append(index)
					index = []
					index.append(j)
				document_index_list.append(document_i_list)
			all_document.append(document_index_list)

		return np.array(all_document)

	def index_document_outproduct(self,document_out_product,query_inner_product):
		temp_matrix = []
		return_matrix = []
		num = self.max_input_query*self.max_input_query

		for i in range(self.max_input_docu):
			temp_matrix = []
			for j in range(self.batch_size):
				document_ii_list = []
				index = []
				index.append(j)
				for k in range(num):
					index.append(i)
					document_ii_list.append(index)
					index = []
					index.append(j)
				temp_matrix.append(document_ii_list)
			return_matrix.append(temp_matrix)
		return np.array(return_matrix)

	def weight_index(self,i_th):
		return_list = []
		weight_indexs = []

		num1 = self.batch_size
		num2 = self.max_input_query*self.max_input_query

		for i in range(num1):
			weight_indexs = []		
			for j in range(num2):
				temp_index = []
				temp_index.append(i)
				temp_index.append(i_th)
				weight_indexs.append(temp_index)
			return_list.append(weight_indexs)
		return np.array(return_list)


	def composite_and_partialTrace(self):
		print ("creat composite and calculate partial trace!!")
		self.document_index_matrix = self.creat_index_matrix(self.batch_size,self.max_input_docu)
		self.query_index_matrix = self.creat_index_matrix(self.batch_size,self.max_input_query)

		self.normal_embedded_q = tf.nn.l2_normalize(self.embedded_chars_q,2)
		self.normal_embedded_d = tf.nn.l2_normalize(self.embedded_chars_d,2)

		#j-th word in query
		for j in range(self.max_input_query):
			query_index = tf.Variable(self.query_index_matrix[j],trainable=False)
			
			j_th_word = tf.expand_dims(tf.gather_nd(self.normal_embedded_q, query_index),-1)

			embedded_chars_q1 = tf.expand_dims(self.normal_embedded_q,-1)
			j_th_word_T = tf.transpose(j_th_word,perm = [0,1,3,2])
			if j == 0:
				self.all_query_dot = tf.matmul(j_th_word_T,embedded_chars_q1)
			else:
				self.all_query_dot = tf.concat([self.all_query_dot,tf.matmul(j_th_word_T,embedded_chars_q1)],1)
		
		#i-th word in document		
		for i in range(self.max_input_docu):
			document_index = tf.Variable(self.document_index_matrix[i],trainable=False)
			
			i_th_word = tf.expand_dims(tf.gather_nd(self.normal_embedded_d, document_index),-1)
			embedded_chars_d1_T = tf.transpose(tf.expand_dims(self.normal_embedded_d,-1),perm = [0,1,3,2])
			d_dT = tf.matmul(i_th_word,embedded_chars_d1_T)#(?,?,50,50)
			
			self.document_outproduct_index = self.index_document_outproduct(d_dT,self.all_query_dot)
			for k in range(self.max_input_docu):
				out_document_index = tf.Variable(self.document_outproduct_index[k],trainable=False)
				
				# muti query
				mul1 = tf.multiply(tf.gather_nd(d_dT, out_document_index), self.all_query_dot)
				
				# multi weight			
				weight_ij = tf.multiply(tf.gather_nd(self.tfidf_value,self.weight_index(i)),tf.gather_nd(self.tfidf_value,self.weight_index(k)))
				
				some_weight = tf.expand_dims(tf.expand_dims(weight_ij,-1),-1)
				
				if k == 0:					
					self.temp_all_composite_matrix = tf.reduce_sum(tf.multiply(mul1,some_weight),1)
				else:					
					self.temp_all_composite_matrix = tf.add(self.temp_all_composite_matrix, tf.reduce_sum(tf.multiply(mul1,some_weight),1))
				
			if i == 0:
				self.reduced_matrix1 =  self.temp_all_composite_matrix
			else:
				self.reduced_matrix1 = tf.add(self.reduced_matrix1,self.temp_all_composite_matrix)

		# self.reduced_matrix = - tf.matmul(self.reduced_matrix1,self.matrix_log())
		self.reduced_matrix = self.reduced_matrix1
		
	def matrix_log(self):
		eigenvalues,eigenvectors = tf.self_adjoint_eig(self.reduced_matrix1)

		eigenvalues_log = tf.log(tf.clip_by_value(eigenvalues,1e-8,tf.reduce_max(eigenvalues)))

		diag_eigenvalues_log = tf.matrix_diag(eigenvalues_log)

		eigenvectors_inverse = tf.matrix_inverse(eigenvectors)
		
		return tf.matmul(tf.matmul(eigenvectors,diag_eigenvalues_log),eigenvectors_inverse)


	def create_loss(self):
		
		self.l2_loss = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.l2_reg_lambda), tf.trainable_variables())

		with tf.name_scope("loss"):
			
			self.p_label = tf.nn.softmax(self.input_label,dim = 0)
			
			cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.p_label, logits=self.logits, dim = 0))
			
			self.loss = tf.clip_by_value(cross_entropy, 1e-8, 2.5) + self.l2_loss

	
	def get_information_from_reduced_matrix(self, input_matrix):
		unit_tensor = tf.eye(input_matrix.get_shape()[1].value, batch_shape=[self.batch_size])
		diag_tensor = tf.matmul(input_matrix,unit_tensor)
		for i in range(self.batch_size):
			single_matrix = input_matrix[i]
			if i == 0:
				self.reduced_matrix_diag = tf.expand_dims(tf.diag_part(single_matrix),0)
				self.reduced_matrix_trace = tf.expand_dims(tf.trace(single_matrix),0)
			else:
				self.reduced_matrix_diag = tf.concat([self.reduced_matrix_diag,tf.expand_dims(tf.diag_part(single_matrix),0)],0)
				self.reduced_matrix_trace = tf.concat([self.reduced_matrix_trace,tf.expand_dims(tf.trace(single_matrix),0)],0)
		return tf.concat([self.reduced_matrix_diag, tf.expand_dims(self.reduced_matrix_trace,1)],1)

	def feed_neural_work_query_gate(self):
		self.reduced_matrix_information = self.get_information_from_reduced_matrix(self.reduced_matrix)
		
		self.layer_input = tf.matmul(self.normal_embedded_q,self.conv2_out)
		self.layer_input = tf.multiply(self.layer_input,self.query_gate_weight)
		self.layer_input = tf.reshape(self.layer_input,[self.batch_size,-1])
		self.layer_input = tf.concat([self.layer_input,self.reduced_matrix_information],1)
		
		print ("MLP layer_input shape : {}".format(self.layer_input.get_shape()))

		layer1 = tf.layers.dense(
				inputs=self.layer_input,
				units=512,
				activation=tf.nn.tanh,
				kernel_initializer=tf.random_normal_initializer(mean = 0, stddev= 0.1 / np.sqrt(512)),
				bias_initializer=tf.constant_initializer(0.1),
				name = "layer1")

		layer2 = tf.layers.dense(
				inputs=layer1,
				units=128,
				activation=tf.nn.tanh,
				kernel_initializer=tf.random_normal_initializer(mean = 0, stddev= 0.1 / np.sqrt(512)),
				bias_initializer=tf.constant_initializer(0.1),
				name = "layer2")

		layer3 = tf.layers.dense(
				inputs=layer2,
				units=64,
				activation=tf.nn.tanh,
				kernel_initializer=tf.random_normal_initializer(mean = 0, stddev= 0.1 / np.sqrt(128)),
				bias_initializer=tf.constant_initializer(0.1),
				name = "layer3")

		self.h_drop = tf.nn.dropout(layer3, self.dropout_keep_prob, name="hidden_output_drop")

		output = tf.layers.dense(
				inputs=self.h_drop,
				units=1,
				activation=tf.nn.tanh,
				kernel_initializer=tf.random_normal_initializer(mean = 0, stddev= 0.1 / np.sqrt(64)),
				bias_initializer=tf.constant_initializer(0.1),
				name = "out_layer")
		self.logits = 3*output
		self.scores = 3*output

	def ngram_cnn_network(self):

		self.CNN_input = tf.expand_dims(self.reduced_matrix,-1)
		with tf.variable_scope("ngram_cnn",reuse=tf.AUTO_REUSE):
			# cnn layer1
			conv_outs = []
			for size in self.kernel_sizes:

				conv_matrix = tf.layers.conv2d(self.CNN_input, 16, size, strides = 1, padding='SAME')				

				max_pool = tf.layers.max_pooling2d(conv_matrix, pool_size=[2,2],strides=[1,1],padding='SAME')
				conv_outs.append(max_pool)
			
			for i in range(len(self.kernel_sizes)):
				if i == 0:
					self.conv_pooling_outs = conv_outs[i]
				else:
					self.conv_pooling_outs = tf.concat([self.conv_pooling_outs, conv_outs[i]],-1)
			
			# cnn layer2
			conv_layer2 = tf.layers.conv2d(self.conv_pooling_outs, 1, 5, strides = 1, padding='SAME')
			self.conv2_max_pool = tf.layers.max_pooling2d(conv_layer2, pool_size=[2,2],strides=[1,1],padding='SAME')
			self.conv2_out = tf.reshape(self.conv2_max_pool,[self.batch_size,self.conv2_max_pool.get_shape()[1].value,-1])
	
                
	def build_graph(self):
		self.creat_placeholder()
		self.load_embeddings()
		self.composite_and_partialTrace()
		self.ngram_cnn_network()
		self.feed_neural_work_query_gate()
		self.create_loss()
		print ("end build graph")
		# exit()


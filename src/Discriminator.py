#coding=utf-8
import tensorflow as tf
import numpy as np
from QACNN import QACNN

class Discriminator(QACNN):
    def __init__(self, sequence_len, batch_size, vocab_size, embedding_size, filter_sizes, num_filters, dropout=1.0, l2_reg=0.0, learning_rate=1e-2, params=None,embeddings=None,loss='log',trainable=True, visible_size, hidden_size):
        QACNN.__init__(self,sequence_len,batch_size,vocab_size,embedding_size,filter_sizes,num_filters,dropout,l2_reg,params,learning_rate,embeddings,loss,trainable)
        self.model_type = 'Dis'
        self.visible_size = visible_size
        self.hidden_size = hidden_size 
        self.d_params = []
        self.d_param.extend(self.updated_params)
        self.pos_prof = tf.placeholder(tf.int32, [None, self.visible_size], name="pos_profile")
        self.neg_prof = tf.placeholder(tf.int32, [None, self.visible_size], name="neg_profile")
        with tf.name_scope('profile_params'):
            if params == None:
                self.W1 = tf.get_variable('weight_1', [self.visible_size, self.hidden_size],
                                         initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                self.Wc1 = tf.get_variable('weight_combined1',[2,2],initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                self.W2 = tf.get_variable('weight_2', [self.hidden_size, 1],
                                         initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                self.Wc2 = tf.get_variable('weight_combined2',[2,1],initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                self.b = tf.get_variable('b', [self.hidden_size], initializer=tf.constant_initializer(0.0))
                self.bc = tf.get_variable('bc', [2], initializer=tf.constant_initializer(0.0))    
            else:
                self.W1 = tf.Variable(params[1][0])
                self.W2 = tf.Variable(params[1][1])
                self.b = tf.Variable(params[1][2])
            d_params.extend([self.W1,self.W2,self.b,self.Wc1,self.Wc2,self.bc])

        pos_prof_score = tf.matmul(tf.nn.tanh(tf.nn.xw_plus_b(self.pos_prof, self.W1, self.b)), self.W2)
        neg_prof_score = tf.matmul(tf.nn.tanh(tf.nn.xw_plus_b(self.neg_prof, self.W1, self.b)), self.W2)        
        
        self.pos_score = tf.matmul(tf.nn.tanh(tf.nn.xw_plus_b([pos_prof_score,self.score12],self.Wc1,self.bc)),self.Wc2)
        self.neg_score = tf.matmul(tf.nn.tanh(tf.nn.xw_plus_b([neg_prof_score,self.score13],self.Wc1,self.bc)),self.Wc2)
        self.l2_loss = tf.constant(0.0)
        for para in self.d_params:
            self.l2_loss+= tf.nn.l2_loss(para)

        with tf.name_scope('output'):
            if loss == 'svm':
                self.losses = tf.maximum(0.0, 0.05 - (self.pos_score - self.neg_score))
                self.loss = tf.reduce_sum(self.losses) + self.l2_reg*self.l2_loss
                self.reward = 2*(tf.sigmoid(0.05 - (self.score12 - self.score13))-0.5)
            elif loss == 'log':
                self.losses = tf.log(tf.sigmoid(self.pos_score - self.neg_score))
                self.loss = -tf.reduce.sum
            self.positive = tf.reduce_mean(self.score12)
            self.negative = tf.reduce_mean(self.score13)
            self.correct = tf.equal(0.0, self.losses)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct,'float'),name='accuracy')

        self.global_step = tf.Variable(0,name='global_step', trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        #optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.updates = optimizer.minimize(self.loss, global_step=self.global_step)

        #grads = optimizer.compute_gradients(self.loss)
        #self.updates = optimizer.apply_gradients(grads, global_step=self.global_step)
        







#coding=utf-8
import tensorflow as tf
import numpy as np
from core import core

class Generator(core):
    def __init__(self, sequence_len, batch_size, vocab_size, embedding_size, filter_sizes, num_filters, visible_size, hidden_size, dropout=1.0, l2_reg=0.0, learning_rate=1e-2, params=None,embeddings=None,loss='svm',trainable=True):
        core.__init__(self,sequence_len,batch_size,vocab_size,embedding_size,filter_sizes,num_filters, visible_size, hidden_size, dropout,l2_reg,params,learning_rate,embeddings,loss,trainable)

        self.model_type = "Gen"
        self.reward  = tf.placeholder(tf.float32, shape=[None], name='reward')
        self.neg_index = tf.placeholder(tf.int32, shape=[None], name='neg_index')

        pos_tmp = tf.reshape([self.pos_prof_score,self.score12], [-1,2])
        neg_tmp = tf.reshape([self.neg_prof_score, self.score13], [-1,2])
        self.pos_score = tf.reshape(tf.matmul(tf.nn.tanh(tf.nn.xw_plus_b(pos_tmp,self.Wc1,self.bc)),self.Wc2),[-1])
        self.neg_score = tf.reshape(tf.matmul(tf.nn.tanh(tf.nn.xw_plus_b(neg_tmp,self.Wc1,self.bc)),self.Wc2),[-1])
        self.gan_score = - tf.abs(self.neg_score - self.pos_score)

        self.batch_scores =tf.nn.softmax(self.gan_score) 
        self.prob = tf.gather(self.batch_scores,self.neg_index)
        self.gan_loss =  -tf.reduce_mean(tf.log(tf.clip_by_value(self.prob,1e-8,tf.reduce_max(self.prob))) *self.reward) +l2_reg* self.l2_loss
        
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.gan_loss)
        self.gan_updates = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        # minize attention
        #self.gan_score=self.neg_score-self.pos_score
        self.dns_score=self.neg_score
        self.positive= tf.reduce_mean(self.pos_score)
        self.negative= tf.reduce_mean(self.neg_score)

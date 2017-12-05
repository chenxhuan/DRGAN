#coding=utf-8
import tensorflow as tf
import numpy as np
from QACNN import QACNN

class Discriminator(QACNN):
    def __init__(self, sequence_len, batch_size, vocab_size, embedding_size, filter_sizes, num_filters, dropout=1.0, l2_reg=0.0, learning_rate=1e-2, params=None,embeddings=None,loss='pair',trainable=True):
        QACNN.__init__(self,sequence_len,batch_size,vocab_size,embedding_size,filter_sizes,num_filters,dropout,l2_reg,params,learning_rate,embeddings,loss,trainable)
        self.model_type = 'Dis'
        with tf.name_scope('output'):
            self.losses = tf.maximum(0.0, tf.subtract(0.05,tf.subtract(self.score12,self.score13)))
            self.loss = tf.reduce_sum(self.losses) + self.l2_reg*self.l2_loss
            self.reward = 2*(tf.sigmoid(tf.subtract(0.05,tf.subtract(self.score12,self.score13)))-0.5)
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




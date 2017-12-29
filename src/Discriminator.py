#coding=utf-8
import tensorflow as tf
import numpy as np
from core import core

class Discriminator(core):
    def __init__(self, sequence_len, batch_size, vocab_size, embedding_size, filter_sizes, num_filters, visible_size, hidden_size, dropout=1.0, l2_reg=0.0, learning_rate=1e-2, params=None,embeddings=None,loss='svm',trainable=True):
        core.__init__(self,sequence_len,batch_size,vocab_size,embedding_size,filter_sizes,num_filters, visible_size, hidden_size, dropout,l2_reg,params,learning_rate,embeddings,loss,trainable)
        self.model_type = 'Dis'
        with tf.name_scope('output'):
            if loss == 'svm':
                self.losses = tf.maximum(0.0, 0.05 - (self.pos_score - self.neg_score))
                self.loss = tf.reduce_sum(self.losses) + self.l2_reg*self.l2_loss
                self.reward = 2*(tf.sigmoid(0.05 - (self.pos_score - self.neg_score))-0.5)
            elif loss == 'log':
                self.losses = tf.log(tf.sigmoid(self.pos_score - self.neg_score))
                self.loss = -tf.reduce_mean(self.losses) + self.l2_reg*self.l2_loss
                self.reward = tf.reshape(tf.log(tf.sigmoid(self.neg_score - self.pos_score)),[-1])

            self.positive = tf.reduce_mean(self.pos_score)
            self.negative = tf.reduce_mean(self.neg_score)
            self.correct = tf.equal(0.0, self.losses)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct,'float'),name='accuracy')

        self.global_step = tf.Variable(0,name='global_step', trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        #optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.updates = optimizer.minimize(self.loss, global_step=self.global_step)

        #grads = optimizer.compute_gradients(self.loss)
        #self.updates = optimizer.apply_gradients(grads, global_step=self.global_step)
        






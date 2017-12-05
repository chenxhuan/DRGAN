#coding=utf-8
'''
Created on 2017-11-29

@author: XinHuan
'''

import tensorflow as tf 
import numpy as np 
import  cPickle,time,sys
sys.path.append('..')
reload(sys)
sys.setdefaultencoding('utf-8')
path = '../model/'


class QACNN():
    
    def __init__(self, sequence_len, batch_size,vocab_size, embedding_size,filter_sizes, num_filters, dropout=1.0,l2_reg=0.0,params=None,learning_rate=1e-2,embeddings=None,loss="pair",trainable=True):
        self.sequence_len=sequence_len
        self.learning_rate=learning_rate
        self.params=params
        self.filter_sizes=filter_sizes
        self.num_filters=num_filters
        self.l2_reg=l2_reg
        self.dropout = dropout
        self.embeddings=embeddings


        self.embedding_size=embedding_size
        self.batch_size=batch_size
        self.model_type="base"
        self.num_filters_total=self.num_filters * len(self.filter_sizes)

        self.input_x_1 = tf.placeholder(tf.int32, [None, sequence_len], name="input_x_1")
        self.input_x_2 = tf.placeholder(tf.int32, [None, sequence_len], name="input_x_2")
        self.input_x_3 = tf.placeholder(tf.int32, [None, sequence_len], name="input_x_3")
        
        self.label=tf.placeholder(tf.float32, [batch_size], name="input_x_3")
        
        # Embedding layer
        self.updated_params=[]
        with tf.name_scope("embedding"):
            if self.params==None:
                if self.embeddings ==None:
                    print ("random embedding")
                    self.Embedding_W = tf.Variable(
                        tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                        name="random_W")
                else:
                    self.Embedding_W = tf.Variable(np.array(self.embeddings),name="embedding_W" ,dtype="float32",trainable=trainable)
            else:
                print ("load embeddings")
                self.Embedding_W=tf.Variable(self.params[0],trainable=trainable,name="embedding_W")
            self.updated_params.append(self.Embedding_W)

        self.kernels=[]        
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, embedding_size, 1, self.num_filters]
                if self.params==None:
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="kernel_W")
                    b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="kernel_b")
                    self.kernels.append((W,b))
                else:
                    _W,_b=self.params[1][i]
                    W=tf.Variable(_W)                
                    b=tf.Variable(_b)
                    self.kernels.append((W,b))   
                self.updated_params.append(W)
                self.updated_params.append(b)

        

        #self.l2_loss = tf.constant(0.0)
        #for para in self.updated_params:
        #    self.l2_loss+= tf.nn.l2_loss(para)
        

        with tf.name_scope("output"):
            q  =self.getRepresentation(self.input_x_1)
            pos=self.getRepresentation(self.input_x_2)
            neg=self.getRepresentation(self.input_x_3)

            self.score12 = self.cosine(q,pos)
            self.score13 = self.cosine(q,neg)

            self.positive= tf.reduce_mean(self.score12)
            self.negative= tf.reduce_mean( self.score13)

    def getRepresentation(self,sentence):
        embedded_chars_1 = tf.nn.embedding_lookup(self.Embedding_W, sentence)
        embedded_chars_expanded_1 = tf.expand_dims(embedded_chars_1, -1)
        output=[]
        for i, filter_size in enumerate(self.filter_sizes): 
            conv = tf.nn.conv2d(
                embedded_chars_expanded_1,
                self.kernels[i][0],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="conv-1"
            )
            h = tf.nn.relu(tf.nn.bias_add(conv, self.kernels[i][1]), name="relu-1")
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, self.sequence_len - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="poll-1"
            )
            output.append(pooled)
        pooled_reshape = tf.reshape(tf.concat(output,3), [-1, self.num_filters_total]) 
        pooled_flat = tf.nn.dropout(pooled_reshape, self.dropout)
        return pooled_flat
    def cosine(self,q,a):

        pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.multiply(q, q), 1)) 
        pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.multiply(a, a), 1))

        pooled_mul_12 = tf.reduce_sum(tf.multiply(q, a), 1) 
        score = tf.div(pooled_mul_12, tf.multiply(pooled_len_1, pooled_len_2)+1e-8, name="scores") 
        return score 
      
    
    
    def save_model(self, sess,precision_current=0):

        now = int(time.time())             
        timeArray = time.localtime(now)
        timeStamp = time.strftime("%Y%m%d%H%M%S", timeArray)
        filename=path+self.model_type+str(precision_current)+"-"+timeStamp+".model"

        param = sess.run([self.Embedding_W,self.kernels])
        cPickle.dump(param, open(filename, 'w'))
        return filename



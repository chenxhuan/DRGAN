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


class core():
    
    def __init__(self, sequence_len, batch_size,vocab_size, embedding_size,filter_sizes, num_filters, query_size, response_size, dropout=1.0,l2_reg=0.0,params=None,learning_rate=1e-2,embeddings=None,loss="pair",trainable=True, score_type = 'nn_output'):
        self.sequence_len=sequence_len
        self.learning_rate=learning_rate
        self.params=params
        self.filter_sizes=filter_sizes
        self.num_filters=num_filters
        self.l2_reg=l2_reg
        self.dropout = dropout
        self.embeddings=embeddings
        self.query_size = query_size
        self.response_size = response_size
        self.neuro_size = self.query_size + self.response_size


        self.embedding_size=embedding_size
        self.batch_size=batch_size
        self.model_type="base"
        self.num_filters_total=self.num_filters * len(self.filter_sizes)

        self.input_x_1 = tf.placeholder(tf.int32, [None, sequence_len], name="input_x_1")
        self.input_x_2 = tf.placeholder(tf.int32, [None, sequence_len], name="input_x_2")
        self.input_x_3 = tf.placeholder(tf.int32, [None, sequence_len], name="input_x_3")
        self.prof_1 = tf.placeholder(tf.float32, [None, self.query_size], name="query_profile")
        self.prof_2 = tf.placeholder(tf.float32, [None, self.response_size], name="response_pos_profile")
        self.prof_3 = tf.placeholder(tf.float32, [None, self.response_size], name="response_neg_profile")
 
        
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

        with tf.name_scope('profile_params'):
            if params == None or len(params) < 3:
                if score_type == 'nn_output':
                    self.W1 = tf.Variable(tf.truncated_normal([self.neuro_size, self.neuro_size], stddev=0.1), name="weight_1")
                    self.Wc1 = tf.Variable(tf.truncated_normal([2,2], stddev=0.1), name="weight_combined1")
                    self.W2 = tf.Variable(tf.truncated_normal([self.neuro_size,1],stddev=0.1), name="weight_2")
                    self.Wc2 = tf.Variable(tf.truncated_normal([2,1],stddev=0.1), name="weight_combined2")
                    self.b = tf.Variable(tf.constant(0.0, shape=[self.neuro_size]), name="b")
                    self.bc = tf.Variable(tf.constant(0.0, shape=[2]), name="bc")
                elif score_type == 'cosine_output':
                    self.W1 = tf.Variable(tf.truncated_normal([self.num_filters_total+self.query_size, self.embedding_size], stddev=0.1), name="weight_1")
                    self.Wc1 = tf.Variable(tf.truncated_normal([self.num_filters_total+self.response_size,self.embedding_size], stddev=0.1), name="weight_combined1")
                    self.W2 = tf.Variable(tf.truncated_normal([self.neuro_size,1],stddev=0.1), name="weight_2")
                    self.Wc2 = tf.Variable(tf.truncated_normal([2,1],stddev=0.1), name="weight_combined2")
                    self.b = tf.Variable(tf.constant(0.0, shape=[self.embedding_size]), name="b")
                    self.bc = tf.Variable(tf.constant(0.0, shape=[self.embedding_size]), name="bc")

            else:
                print np.shape(params[2][0])
                self.W1 = tf.Variable(params[2][0])
                self.W2 = tf.Variable(params[2][1])
                self.b = tf.Variable(params[2][2])
                self.Wc1 = tf.Variable(params[2][3])
                self.Wc2 = tf.Variable(params[2][4])
                self.bc = tf.Variable(params[2][5])
            self.updated_params.extend([self.W1,self.W2,self.b,self.Wc1,self.Wc2,self.bc])



        self.l2_loss = tf.constant(0.0)
        for para in self.updated_params:
            self.l2_loss+= tf.nn.l2_loss(para)
        

        with tf.name_scope("output"):
            q  =self.getRepresentation(self.input_x_1)
            pos=self.getRepresentation(self.input_x_2)
            neg=self.getRepresentation(self.input_x_3)
            if score_type == 'nn_output':
                self.score12 = self.cosine(q,pos)
                self.score13 = self.cosine(q,neg)

                self.pos_prof_score = tf.reshape(tf.matmul(tf.nn.tanh(tf.nn.xw_plus_b(tf.concat([self.prof_1,self.prof_2],1), self.W1, self.b)), self.W2),[-1])
                self.neg_prof_score = tf.reshape(tf.matmul(tf.nn.tanh(tf.nn.xw_plus_b(tf.concat([self.prof_1,self.prof_3],1), self.W1, self.b)), self.W2),[-1])
          
                pos_tmp = tf.reshape([self.pos_prof_score,self.score12], [-1,2])
                neg_tmp = tf.reshape([self.neg_prof_score, self.score13], [-1,2])
                self.pos_score = tf.reshape(tf.nn.softplus(tf.matmul(tf.nn.tanh(tf.nn.xw_plus_b(pos_tmp,self.Wc1,self.bc)),self.Wc2)),[-1])
                self.neg_score = tf.reshape(tf.nn.softplus(tf.matmul(tf.nn.tanh(tf.nn.xw_plus_b(neg_tmp,self.Wc1,self.bc)),self.Wc2)),[-1])
            elif score_type == 'cosine_output':
                print score_type
                query = tf.concat([self.prof_1,q],1)
                pos_resp = tf.concat([self.prof_2,pos],1)
                neg_resp = tf.concat([self.prof_3,neg],1)
                query_tmp = tf.nn.tanh(tf.nn.xw_plus_b(query,self.W1,self.b))
                pos_tmp = tf.nn.tanh(tf.nn.xw_plus_b(pos_resp,self.Wc1,self.bc))
                neg_tmp = tf.nn.tanh(tf.nn.xw_plus_b(neg_resp,self.Wc1,self.bc))
                self.pos_score = self.cosine(query_tmp,pos_tmp)
                self.neg_score = self.cosine(query_tmp,neg_tmp)


    def getRepresentation(self,sentence):
        embedded_chars_1 = tf.nn.embedding_lookup(self.Embedding_W, sentence)
        embedded_chars_expanded_1 = tf.expand_dims(embedded_chars_1, -1)  # expand the channels of input = 1
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

        param = sess.run([self.Embedding_W,self.kernels,[self.W1,self.W2,self.b,self.Wc1,self.Wc2,self.bc]])
        cPickle.dump(param, open(filename, 'w'))
        return filename



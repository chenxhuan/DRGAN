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
    
    def __init__(self, sequence_len, batch_size,vocab_size, embedding_size,filter_sizes, num_filters, visible_size, hidden_size, dropout=1.0,l2_reg=0.0,params=None,learning_rate=1e-2,embeddings=None,loss="pair",trainable=True):
        self.sequence_len=sequence_len
        self.learning_rate=learning_rate
        self.params=params
        self.filter_sizes=filter_sizes
        self.num_filters=num_filters
        self.l2_reg=l2_reg
        self.dropout = dropout
        self.embeddings=embeddings
        self.visible_size = visible_size
        self.hidden_size = hidden_size



        self.embedding_size=embedding_size
        self.batch_size=batch_size
        self.model_type="base"
        self.num_filters_total=self.num_filters * len(self.filter_sizes)

        self.input_x_1 = tf.placeholder(tf.int32, [None, sequence_len], name="input_x_1")
        self.input_x_2 = tf.placeholder(tf.int32, [None, sequence_len], name="input_x_2")
        self.input_x_3 = tf.placeholder(tf.int32, [None, sequence_len], name="input_x_3")
        self.pos_prof = tf.placeholder(tf.float32, [None, self.visible_size], name="pos_profile")
        self.neg_prof = tf.placeholder(tf.float32, [None, self.visible_size], name="neg_profile")
 
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

        with tf.name_scope('profile_params'):
            if params == None or len(params) < 3:
                self.W1 = tf.Variable(tf.truncated_normal([self.visible_size, self.hidden_size], stddev=0.1), name="weight_1")
                self.Wc1 = tf.Variable(tf.truncated_normal([2,2], stddev=0.1), name="weight_combined1")
                self.W2 = tf.Variable(tf.truncated_normal([self.hidden_size,1],stddev=0.1), name="weight_2")
                self.Wc2 = tf.Variable(tf.truncated_normal([2,1],stddev=0.1), name="weight_combined2")
                self.b = tf.Variable(tf.constant(0.0, shape=[self.hidden_size]), name="b")
                self.bc = tf.Variable(tf.constant(0.0, shape=[2]), name="bc")
                #self.lamda = tf.Variable(tf.constant(0.1, shape=[1]), name="lamda_weight")

                #self.W1 = tf.Variable(name='weight_1', [self.visible_size, self.hidden_size],initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                #self.Wc1 = tf.get_variable('weight_combined1',[2,2],initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                #self.W2 = tf.get_variable('weight_2', [self.hidden_size, 1],initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                #self.Wc2 = tf.get_variable('weight_combined2',[2,1],initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                #self.b = tf.get_variable('b', [self.hidden_size], initializer=tf.constant_initializer(0.0))
                #self.bc = tf.get_variable('bc', [2], initializer=tf.constant_initializer(0.0))
            else:
                self.W1 = tf.Variable(params[2][0])
                self.W2 = tf.Variable(params[2][1])
                self.b = tf.Variable(params[2][2])
                self.Wc1 = tf.Variable(params[2][3])
                self.Wc2 = tf.Variable(params[2][4])
                self.bc = tf.Variable(params[2][5])
                #self.lamda = tf.Variable(params[2][6])
            self.updated_params.extend([self.W1,self.W2,self.b,self.Wc1,self.Wc2,self.bc])



        self.l2_loss = tf.constant(0.0)
        for para in self.updated_params:
            self.l2_loss+= tf.nn.l2_loss(para)
        

        with tf.name_scope("output"):
            q  =self.getRepresentation(self.input_x_1)
            pos=self.getRepresentation(self.input_x_2)
            neg=self.getRepresentation(self.input_x_3)

            self.score12 = self.cosine(q,pos)
            self.score13 = self.cosine(q,neg)

            self.pos_prof_score = tf.reshape(tf.matmul(tf.nn.tanh(tf.nn.xw_plus_b(self.pos_prof, self.W1, self.b)), self.W2),[-1])
            self.neg_prof_score = tf.reshape(tf.matmul(tf.nn.tanh(tf.nn.xw_plus_b(self.neg_prof, self.W1, self.b)), self.W2),[-1])
            self.combined_score = self.score12 + self.pos_prof_score
          
            pos_tmp = tf.reshape([self.pos_prof_score,self.score12], [-1,2])
            neg_tmp = tf.reshape([self.neg_prof_score, self.score13], [-1,2])
            self.pos_score = tf.reshape(tf.nn.tanh(tf.matmul(tf.nn.tanh(tf.nn.xw_plus_b(pos_tmp,self.Wc1,self.bc)),self.Wc2)),[-1])
            self.neg_score = tf.reshape(tf.nn.tanh(tf.matmul(tf.nn.tanh(tf.nn.xw_plus_b(neg_tmp,self.Wc1,self.bc)),self.Wc2)),[-1])
            #self.pos_score = tf.reshape(tf.nn.relu(tf.nn.xw_plus_b(pos_tmp,self.Wc2,self.bc)),[-1])
            #self.neg_score = tf.reshape(tf.nn.relu(tf.nn.xw_plus_b(neg_tmp,self.Wc2,self.bc)),[-1])
            #self.pos_score = self.lamda*self.score12 + (1-self.lamda)*self.pos_prof_score
            #self.neg_score = self.lamda*self.score13 + (1-self.lamda)*self.neg_prof_score

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

        param = sess.run([self.Embedding_W,self.kernels,[self.W1,self.W2,self.b,self.Wc1,self.Wc2,self.bc]])
        cPickle.dump(param, open(filename, 'w'))
        return filename



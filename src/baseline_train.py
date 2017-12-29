import tensorflow as tf
import time,math, os, sys, datetime, random, cPickle,pdb
import numpy as np
import Discriminator
from dataPrepare import generate_uniform_pair, generate_test_samples
from util import evaluation
sys.path.append('..')
reload(sys)
sys.setdefaultencoding('utf-8')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

path = '../data/'
timestamp = lambda : time.strftime('%Y%m%d%H%M%S', time.localtime(int(time.time())))
precision_log = 'log/'+timestamp()+'test_baseline_dns.log'
pre_trained_path = '../model/'

tf.flags.DEFINE_integer("max_sequence_len", 200, "Max sequence length fo sentence (default: 200)")
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 500, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout", 1.0, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg", 0.001, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 0.1, "learning_rate (default: 0.1)")
# Training parameters
tf.flags.DEFINE_integer("batch_size", 80, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("pools_size", 100, "The sampled set of a positive ample, which is bigger than 500")
tf.flags.DEFINE_integer("sampled_size", 100, " the real selectd set from the The sampled pools")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
vocab = cPickle.load(open(path+'vocab','r'))

def generate_dns_pair(sess,model, eb_samples, pro_samples):
    eb_tmp, pro_tmp = [],[]
    batch_size = FLAGS.batch_size
    for _index, row in enumerate(eb_samples):
    	candidate_index = range(len(eb_samples))
    	sampled_index = list(np.random.choice(candidate_index, size=[batch_size],replace=False))
        eb_sample_pools = eb_samples[sampled_index]
        pro_sample_pools = pro_samples[sampled_index]
        feed_dict  = {
        	model.input_x_1:[row[0]]*batch_size,
                model.input_x_2:[row[1]]*batch_size,
                model.input_x_3:eb_sample_pools[:,2],
                model.pos_prof:[pro_samples[_index][0]]*batch_size,
                model.neg_prof:pro_sample_pools[:,1]}
        predicted = sess.run(model.neg_score, feed_dict)
	neg_index = np.argmax(predicted)
        eb_tmp.append([row[0],row[1],eb_samples[neg_index][2]])
        pro_tmp.append([pro_samples[_index][0], pro_samples[neg_index][1]])
    return np.array(eb_tmp), np.array(pro_tmp)

 

def process():
    with tf.Graph().as_default():
        #with tf.device('/gpu:1'):
            session_conf = tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement
            )
            sess = tf.Session(config=session_conf)
            with sess.as_default(), open(precision_log,'w') as log:
                if len(sys.argv) > 1:
                    param = cPickle.load(open(pre_trained_path+sys.argv[1],'r'))
                    print param[2][3],param[2][4],param[2][5]
                else:
                    param = None
                #samples = generate_uniform_pair('test_feature')
                #eb_samples, pro_samples = generate_uniform_pair('train_feature')
                eb_samples, pro_samples = cPickle.load(open(path+'train_samples','r'))
                profile_dim = len(pro_samples[0][-1])
                batch_size = FLAGS.batch_size
                num_batches = int(math.ceil(len(eb_samples)/batch_size))
                print np.shape(eb_samples),np.shape(pro_samples), profile_dim
                dis = Discriminator.Discriminator(
                    FLAGS.max_sequence_len,
                    FLAGS.batch_size,
                    len(vocab),
                    FLAGS.embedding_dim,
                    list(map(int, FLAGS.filter_sizes.split(","))),
                    FLAGS.num_filters,
                    profile_dim,
                    profile_dim,
                    FLAGS.dropout,
                    FLAGS.l2_reg,
                    FLAGS.learning_rate,
                    param,
                    None,
                    'svm',
                    True) 
                sess.run(tf.global_variables_initializer())
                for i in range(FLAGS.num_epochs):
                    step, current_loss, accuracy = 0, 0.0, 0.0
		    eb_samples, pro_samples = generate_dns_pair(sess,dis,eb_samples,pro_samples)
                    for ib in range(num_batches):
                        end_index = min((ib+1)*batch_size, len(eb_samples))
                        eb_batch = eb_samples[end_index-batch_size:end_index]
                        pro_batch = pro_samples[end_index-batch_size:end_index]
                        feed_dict = { 
                            dis.input_x_1:eb_batch[:,0],
                            dis.input_x_2:eb_batch[:,1],
                            dis.input_x_3:eb_batch[:,2],
                            dis.pos_prof:pro_batch[:,0],
                            dis.neg_prof:pro_batch[:,1]}
                        _,step, current_loss, accuracy,positive = sess.run([dis.updates,dis.global_step,dis.loss,dis.accuracy,dis.pos_score],feed_dict)
                    line = ("%s: basic Dis step %d, loss %f with acc %f,total step: %d "%(timestamp(), step, current_loss,accuracy,FLAGS.num_epochs*num_batches))
                    print line
                    log.write(line+"\n")
                    if i != FLAGS.num_epochs-1:
                        evaluation(sess,dis,log,batch_size)
                evaluation(sess,dis,log,batch_size,True)
if __name__ == '__main__':
    start = time.time()
    process()
    end = time.time()
    print 'training time',(end - start)/60


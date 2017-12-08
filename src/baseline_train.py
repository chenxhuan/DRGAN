import tensorflow as tf
import time,math, os, sys, datetime, random, cPickle,pdb
import numpy as np
import Discriminator
from dataPrepare import generate_uniform_pair, generate_test_samples

sys.path.append('..')
reload(sys)
sys.setdefaultencoding('utf-8')
path = '../data/'
timestamp = lambda : time.strftime('%Y%m%d%H%M%S', time.localtime(int(time.time())))
precision_log = 'log/'+timestamp()+'test_dns.log'


tf.flags.DEFINE_integer("max_sequence_len", 200, "Max sequence length fo sentence (default: 200)")
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 500, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout", 1.0, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg", 0.001, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 0.1, "learning_rate (default: 0.1)")
# Training parameters
tf.flags.DEFINE_integer("batch_size", 80, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")
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

def evaluation(sess,model,log):
    asker_eva = []
    current_step = tf.train.global_step(sess, model.global_step)
    batch_size = FLAGS.batch_size
    eb_samples,pro_samples, asker_label = generate_test_samples()
    num_batches = int(math.ceil(len(eb_samples)/batch_size))
    batch_scores, asker_scores = [],{}
    for i in range(num_batches):
        end_index = min((i+1)*batch_size, len(eb_samples))
        eb_batch = eb_samples[end_index-batch_size:end_index]
        pro_batch = pro_samples[end_index-batch_size:end_index]
        feed_dict = {
            model.input_x_1:eb_batch[:,0],
            model.input_x_2:eb_batch[:,1],
            model.input_x_3:eb_batch[:,2],
            model.pos_prof:pro_batch[:,0],
            model.neg_prof:pro_batch[:,1]} 
        batch_scores.extend(sess.run(model.pos_score, feed_dict))
    print len(batch_scores)
    for index, s in enumerate(batch_scores):
        key, label = asker_label[index]
        scoreList = asker_scores.get(key,[])
        scoreList.append((s,label))
        asker_scores[key] = scoreList
    print len(asker_scores),asker_scores.values()[0]
    for key, s_l in asker_scores.items():
        mscore = sorted(s_l,key=lambda x:x[0], reverse=True)
        max_score = mscore[0][0]
        flag = 0
        for s,l in mscore:
            if s < max_score:
                break
            elif l == '1.0':
                flag = 1
        asker_eva.append(flag)
    accuracy = sum(asker_eva)*1.0 / len(asker_eva)
    line="%d samples test: %d epoch: precision %f"%(len(asker_scores),current_step,accuracy)
    print (line)
    print( model.save_model(sess,accuracy))
    log.write(line+"\n")


def process():
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            session_conf = tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement
            )
            sess = tf.Session(config=session_conf)
            with sess.as_default(), open(precision_log,'w') as log:
                param = None
                #samples = generate_uniform_pair('test_feature')
                eb_samples, pro_samples = generate_uniform_pair('train_feature')
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
                    print(("%s: basic Dis step %d, loss %f with acc %f,total step: %d "%(timestamp(), step, current_loss,accuracy,FLAGS.num_epochs*num_batches)))
                evaluation(sess,dis,log)
if __name__ == '__main__':
    start = time.time()
    process()
    end = time.time()
    print 'training time',(end - start)/60


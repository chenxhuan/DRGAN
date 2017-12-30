#coding=utf-8
import sys, cPickle,math
import numpy as np
import tensorflow as tf
sys.path.append('..')
reload(sys)
sys.setdefaultencoding('utf-8')
path = '../data/'

def precision_at_k(r,k):
    assert k >= 1
    return np.mean(r[:k])
def average_precision(r):
    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)
def MAP(rs):
    return np.mean([average_precision(r) for r in rs])
def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))
def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

def metrics(l):
    p1 = precision_at_k(l,1)
    p3 = precision_at_k(l,3)

    ndcg1 = ndcg_at_k(l,1)
    ndcg3 = ndcg_at_k(l,3)
    return np.array([p1,p3,ndcg1,ndcg3])

def evaluation(sess,model,log,batch_size, save_flag=False):
    asker_MAP = []
    result = np.array([0.0]*4)
    current_step = tf.train.global_step(sess, model.global_step)
    #eb_samples,pro_samples, asker_label = generate_test_samples()
    eb_samples,pro_samples, asker_label = cPickle.load(open(path+'test_samples','r'))
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
            model.prof_1:list(pro_batch[:,0]),
            model.prof_2:list(pro_batch[:,1]),
            model.prof_3:list(pro_batch[:,2])}
        batch_scores.extend(sess.run(model.pos_score, feed_dict))
    print len(batch_scores)
    for index, s in enumerate(batch_scores):
        key, label = asker_label[index]
        scoreList = asker_scores.get(key,[])
        scoreList.append((s,label))
        asker_scores[key] = scoreList
    print len(asker_scores)
    for i, (k, s_l) in enumerate(asker_scores.items()):
        mscore = sorted(s_l,key=lambda x:x[0], reverse=True)
        if  i== 100:
            print mscore
        flag = []
        for s,l in mscore:
            if l == '1.0':
                flag.append(1)
            else:
                flag.append(0)
        result += metrics(flag)
        asker_MAP.append(flag)
    total_MAP = MAP(asker_MAP)
    total_metrics = result / len(asker_scores)
    line="%d samples test: %d epoch, p1,p3,NDCG1,NDCG3,MAP: %f,%f,%f,%f,%f"%(len(asker_scores),current_step,total_metrics[0],total_metrics[1],total_metrics[2],total_metrics[3],total_MAP)
    print (line)
    if save_flag:
        print( model.save_model(sess,total_metrics[0]))
    log.write(line+"\n")
    log.flush()
    #tf.summary.FileWriter(path+'TB/',sess.graph)

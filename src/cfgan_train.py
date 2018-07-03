import tensorflow as tf
from cf.dis_model import DIS
from cf.gen_model import GEN
import cPickle, sys, datetime,time,os
import numpy as np
import multiprocessing
sys.path.append('..')
reload(sys)
sys.setdefaultencoding('utf-8')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

timestamp = lambda : time.strftime('%Y%m%d%H%M%S', time.localtime(int(time.time())))
cores = multiprocessing.cpu_count()

#########################################################################################
# Hyper-parameters
#########################################################################################
EMB_DIM = 100
USER_NUM = 13105
#USER_NUM = 12255
ITEM_NUM = 11648
#ITEM_NUM = 4644
BATCH_SIZE = 80
INIT_DELTA = 0.01

all_items = set(range(ITEM_NUM))
#workdir = '../ask120_data/'
workdir = '../xywy_data/'
DIS_TRAIN_FILE = workdir + 'train_cf4'
DIS_TEST_FILE = workdir + 'test_cf4'

#########################################################################################
# Load data
#########################################################################################
user_pos_train = {}
train_data = []
with open(DIS_TRAIN_FILE)as fin:
    user, item, label = [], [], []
    for line in fin:
        line = line.split()
        uid = int(line[0])
        iid = int(line[1])
        r = float(line[2])
        if r > 0.0:
            if uid in user_pos_train:
                user_pos_train[uid].append(iid)
            else:
                user_pos_train[uid] = [iid]
        

user_pos_test = {}
user_test = {}
with open(DIS_TEST_FILE)as fin:
    for line in fin:
        line = line.split()
        uid = int(line[0])
        iid = int(line[1])
        r = float(line[2])
        if r > 0.0:
            if uid in user_pos_test:
                user_pos_test[uid].append(iid)
            else:
                user_pos_test[uid] = [iid]
        if uid in user_test:
            user_test[uid].append(iid)
        else:
            user_test[uid] = [iid]

all_users = user_pos_train.keys()
all_users.sort()


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))


def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

def precision_at_k(r,k):
    assert k >= 1
    return np.mean(r[:k])


def simple_test_one_user(x):
    rating = x[0]
    u = x[1]
    #print np.shape(rating), u
    #test_items = list(all_items - set(user_pos_train.get(u,[])))
    #test_items = list(all_items)
    test_items = user_test.get(u,[])
    candidate_items = list(np.random.choice(list(all_items - set(test_items)) , size=10-len(test_items) ,replace=False))
    test_items += candidate_items 
    item_score = []
    for i in test_items:
        item_score.append((i, rating[i]))

    item_score = sorted(item_score, key=lambda x: x[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test[u]:
            r.append(1)
        else:
            r.append(0)

    p_1 = precision_at_k(r,1)
    p_3 = precision_at_k(r,3)
    ndcg_1 = ndcg_at_k(r, 1)
    ndcg_3 = ndcg_at_k(r, 3)
    mp = np.mean([precision_at_k(r, k + 1) for k in range(len(r)) if r[k]])

    return np.array([p_1, p_3, ndcg_1, ndcg_3, mp])


def simple_test(sess, model):
    result = np.array([0.] * 5)
    pool = multiprocessing.Pool(cores)
    batch_size = 80
    test_users = user_pos_test.keys()
    test_user_num = len(test_users)
    index = 0
    while True:
        if index >= test_user_num:
            break
        user_batch = test_users[index:index + batch_size]
        index += batch_size
        user_batch_rating = sess.run(model.all_rating, {model.u: user_batch})
        user_batch_rating_uid = zip(user_batch_rating, user_batch)
        batch_result = pool.map(simple_test_one_user, user_batch_rating_uid)
        #for rating, uid in user_batch_rating_uid:
        #    result += simple_test_one_user((rating,uid))
        for re in batch_result:
            result += re

    pool.close()
    ret = result / test_user_num
    ret = [str(x) for x in ret]
    return ret


def generate_for_d(sess, model):
    for u in user_pos_train:
        pos = user_pos_train[u]

        rating = sess.run(model.all_rating, {model.u: [u]})
        rating = np.array(rating[0]) 
        exp_rating = np.exp(rating)
        prob = exp_rating / np.sum(exp_rating)

        neg = np.random.choice(np.arange(ITEM_NUM), size=len(pos), p=prob)
        for i in range(len(pos)):
            train_data.append([u,pos[i],neg[i]])

def get_batch_data(index, size):  
    user = []
    item = []
    label = []
    for i in range(index, index + size):
        line = train_data[i]
        user.append(int(line[0]))
        user.append(int(line[0]))
        item.append(int(line[1]))
        item.append(int(line[2]))
        label.append(1.)
        label.append(0.)
    return user, item, label


def main():
    print "load model..."
    param = None
    generator = GEN(ITEM_NUM, USER_NUM, EMB_DIM, lamda=0.001, param=None, initdelta=INIT_DELTA,
                    learning_rate=0.05)
    discriminator = DIS(ITEM_NUM, USER_NUM, EMB_DIM, lamda=0.001, param=None, initdelta=INIT_DELTA,
                        learning_rate=0.05)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    gen_log = open('log/'+timestamp()+'cfgan_log.txt', 'w')

    # minimax training
    best = 1.
    for epoch in range(20):
            if epoch > 0:
                for d_epoch in range(10):
                    current_loss,logits = 0.0,0.0
                    if d_epoch  == 0:
                        generate_for_d(sess, generator)
                        train_size = len(train_data)
                    index = 1
                    while True:
                        if index > train_size:
                            break
                        if index + BATCH_SIZE <= train_size - 1:
                            input_user, input_item, input_label = get_batch_data(index, BATCH_SIZE)
                        else:
                            input_user, input_item, input_label = get_batch_data(index,train_size - index )
                        index += BATCH_SIZE

                        _, current_loss = sess.run([discriminator.d_updates,discriminator.loss],
                                 feed_dict={discriminator.u: input_user, discriminator.i: input_item,
                                            discriminator.label: input_label})
                    line=("%s: DIS step %d, loss %f  "%(datetime.datetime.now().isoformat(), epoch, current_loss))
                    print line
                result = simple_test(sess, discriminator)
                print "epoch ", epoch, "dis: ", result
                buf = '\t'.join(result)
                gen_log.write(buf+'\t'+line +'\n')
                gen_log.flush()
            # Train G
            for g_epoch in range(5):  # 50
                g_current_loss = 0.0
                for u in user_pos_train:
                    sample_lambda = 0.2
                    pos = user_pos_train[u]

                    rating = sess.run(generator.all_logits, {generator.u: u})
                    exp_rating = np.exp(rating)
                    prob = exp_rating / np.sum(exp_rating)  # prob is generator distribution p_\theta

                    pn = (1 - sample_lambda) * prob
                    pn[pos] += sample_lambda * 1.0 / len(pos)
                    pn /= pn.sum()
                    # Now, pn is the Pn in importance sampling, prob is generator distribution p_\theta
                    sample = np.random.choice(np.arange(ITEM_NUM), BATCH_SIZE, p=pn)
                    ###########################################################################
                    # Get reward and adapt it with importance sampling
                    ###########################################################################
                    reward = sess.run(discriminator.reward, {discriminator.u: u, discriminator.i: sample})
                    reward = reward * prob[sample] / pn[sample]
                    ###########################################################################
                    # Update G
                    ###########################################################################
                    _,g_current_loss = sess.run([generator.gan_updates,generator.gan_loss],
                                 {generator.u: u, generator.i: sample, generator.reward: reward})
                line=("%s: GEN step %d, loss %f  "%(datetime.datetime.now().isoformat(), epoch, g_current_loss)) 
                print line
            result = simple_test(sess, generator)
            print "epoch ", epoch, "gen: ", result
            buf = '\t'.join(result)
            gen_log.write(buf + '\t' + line+'\n')
            gen_log.flush()

            p_5 = result[1]
            if p_5 > best:
                print 'best: ', result
                best = p_5
                #generator.save_model(sess, "../model/cf_gan_generator.pkl")

    gen_log.close()


if __name__ == '__main__':
    main()


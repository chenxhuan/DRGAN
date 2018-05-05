#coding=utf-8
'''
Created on 2017-11-29

@author: XinHuan
'''
import xlrd, jieba, csv, sys, pdb, codecs, random, math, re, cPickle, random
import numpy as np
sys.path.append('..')
reload(sys)
sys.setdefaultencoding('utf-8')
r = "[A-Za-z0-9\[+.!/_,$%^*(+\"\']+|[+——！＼，,。﹌“·”《’‘》;…:：；℅～、~@#￥%……&*（）】【？?\n\t]+"
noUsedWords = {}
path = '../xywy_data/'
max_sequence_len = 200
word_len = []

def extract_data_with_best_answer1(outfile):
    data = xlrd.open_workbook(path+'xywy_origin.xlsx').sheet_by_index(0)
    nrows, ncols = data.nrows, data.ncols
    dic_data = {}
    for index in range(1,nrows):
        if dic_data.has_key(data.cell(index,0).value):
            idBest = dic_data[data.cell(index,0).value].get(data.cell(index,20).value, 0)
            dic_data[data.cell(index,0).value][data.cell(index,20).value] = idBest + 1
        else:
            dic_data[data.cell(index,0).value] = {data.cell(index,20).value:1}
    print len(dic_data)
    count = 0
    gender_dic, age_dic,  doc_title_dic, users_dic, docs_dic = {}, {}, {}, {}, {}
    with open(path+outfile, 'wb') as csvFile:
        outWriter = csv.writer(csvFile)
        for index in range(1,nrows):
            if len(dic_data[data.cell(index,0).value]) > 1:
                count = count + 1
                #pdb.set_trace()
                feature = []
                rows = data.row_values(index)
                feature.append(str(rows[1]).strip())
                if feature[-1] not in users_dic:
                    users_dic[feature[-1]] = len(users_dic)
                if len(rows[2].strip()) == 0:
                    feature.append('unknown')
                    if 'unknown' not in gender_dic:
                        gender_dic['unknown'] = len(gender_dic)
                else:
                    feature.append(str(rows[2].strip()))
                    if str(rows[2].strip()) not in gender_dic:
                        gender_dic[str(rows[2].strip())] = len(gender_dic)
                if len(str(rows[3]).strip()) == 0:
                    feature.append('unknown')
                    if 'unknown' not in age_dic:
                        age_dic['unknown'] = len(age_dic)
                else:
                    feature.append(str(rows[3]).strip())
                    if str(rows[3]).strip() not in age_dic:
                        age_dic[str(rows[3]).strip()] = len(age_dic) 
                feature.append(str(rows[5]).strip()+' '+str(rows[6]).strip())
                feature.append(str(rows[13]).strip())
                if feature[-1] not in docs_dic:
                    docs_dic[feature[-1]] = len(docs_dic)
                if len(rows[14].strip()) == 0:
                    feature.append('unknown')
                    if 'unknown' not in doc_title_dic:
                        doc_title_dic['unknown'] = len(doc_title_dic)
                else:
                    doc_title = str(rows[14].strip())
                    feature.append(doc_title)
                    if doc_title not in doc_title_dic:
                        doc_title_dic[doc_title] = len(doc_title_dic)
                feature.append(str(rows[15]).strip())
                feature.append(str(rows[16]).strip())
                feature.append(str(rows[19]).strip())
                feature.append(str(rows[17]).strip())
                feature.append(rows[20])
                outWriter.writerow(feature)
    print count,  len(gender_dic),len(age_dic),len(doc_title_dic), len(users_dic), len(docs_dic)
    cPickle.dump((gender_dic,age_dic,doc_title_dic,users_dic,docs_dic),open(path+'features.dic','w'))

def init_word_seg():
    with codecs.open("../ask120_data/user.dic","r") as input_userDic:
        userDic = input_userDic.readlines()
        for word in userDic:
            word = word.strip()
            jieba.add_word(str(word))
    with codecs.open("../ask120_data/stopword-full.dic","r") as input_stopwords:
        for index, word in enumerate(input_stopwords.readlines()):
            word = word.strip()
            noUsedWords[str(word)] = index
    
def feature_process2(inFile,outFile):
    init_word_seg()
    gender_dic,age_dic,doc_title_dic,users_dic,docs_dic = cPickle.load(open(path+'features.dic','r'))
    dics_tuple = (users_dic,gender_dic,age_dic,0,docs_dic,doc_title_dic)
    output = codecs.open(path+outFile,'w')
    code = 0
    vocab = {'unknown':code}
    with open(path+inFile,'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            if len(row) < 11:
                continue
            res_line = ''
            for index, cell in enumerate(row):
                if index == 3 or index == 6 or index == 9:
                    segs = jieba.cut(re.sub(r.decode('utf-8'), ' '.decode('utf-8'), cell.decode('utf-8')))
                    for seg in segs:
                        seg = seg.strip()
                        if seg.encode('utf-8') not in noUsedWords and len(seg) > 0:
                            res_line += seg + ' '
                            if seg not in vocab:
                                code += 1
                                vocab[seg] = code
                    res_line = res_line+ '\t'
                elif index == 7 or index == 8 or index == 10:
                    res_line += str(cell)+'\t'
                else:
                    res_line += str(dics_tuple[index][str(cell).strip()])+'\t'
            output.write(res_line.strip()+'\n')
    output.close()
    cPickle.dump(vocab, open(path+'vocab','w'))
    print len(vocab), len(users_dic), len(docs_dic)

def format_str(text, vocab):
    segs = text.strip().split(' ')
    word_len.append(len(segs))
    fill = max(0, max_sequence_len-len(segs))
    segs.extend(['unknown']*fill)
    formatSeg = [] 
    for i in xrange(max_sequence_len):
        if segs[i].decode('utf-8') in vocab:
            formatSeg.append(str(vocab[segs[i].decode('utf-8')]))
        else:
            formatSeg.append(str(vocab['unknown']))
    return '_'.join(formatSeg)
                         
def split_train_test3(inFile):
    asker_map, asker_right, responses = {}, {},[]
    vocab = cPickle.load(open(path+'vocab','r'))
    with codecs.open(path+inFile,'r') as raw_feature:
        lines = raw_feature.readlines() 
        for index,line in enumerate(lines):
            token = line.strip().split('\t')
            if len(token) < 11:
                continue
            key, value = token[:3], token[4:6]
            key.append(format_str(token[3], vocab))
            value.extend(token[7:9])
            value.append(format_str(token[6]+'unknown '+token[9],vocab))
            value.append(token[10])
            skey = '\t'.join(key) 
            svalue = '\t'.join(value)
            responses.append(svalue)
            if skey not in asker_map:
                asker_map[skey] = [svalue]
                if token[10] == '1.0':
                    asker_right[skey] = 1
                else:
                    asker_right[skey] = 0
            else:
                asker_map[skey].append(svalue)
                if token[10] == '1.0':
                    asker_right[skey] += 1
    asker_size = len(asker_map)
    train_out = codecs.open(path+'train_feature','w')
    test_out = codecs.open(path+'test_feature','w')
    train_cf = codecs.open(path+'train_cf','w')
    test_cf = codecs.open(path+'test_cf','w')
    maxsize, minsize = 0, asker_size
    for index,kv in enumerate(asker_map.items()):
        key, values = kv
        maxsize = max(maxsize, len(values))
        minsize = min(minsize, len(values))
        if index < asker_size*0.8:   # train set
            for value in values:
                train_out.write(key+':VS:'+value+'\n')
                cf_value = value.split('\t')
                train_cf.write(key.split('\t')[0]+'\t'+cf_value[0]+'\t'+cf_value[-1]+'\n')
        else:                       # test set
            for value in values:   
                test_out.write(key+':VS:'+value+'\n')
                cf_value = value.split('\t')
                test_cf.write(key.split('\t')[0]+'\t'+cf_value[0]+'\t'+cf_value[-1]+'\n') 
    print asker_size, maxsize, minsize, max(word_len),min(word_len),np.average(word_len)
    train_out.close()
    test_out.close()
    train_cf.close()
    test_cf.close()
    tmp = asker_right.values()
    count = 0
    for i in tmp:
        if i > 1:
            count += 1
    responses = list(set(responses))
    print count,max(tmp),np.average(tmp),len(responses)
    cPickle.dump(asker_map,open(path+'asker_map','w'))
    cPickle.dump(responses,open(path+'responses','w'))

def generate_uniform_pair(dataset):
    embedding_samples, profile_samples = [],[]
    asker_map = cPickle.load(open(path+'asker_map','r'))
    responses = cPickle.load(open(path+'responses','r'))
    print len(asker_map)
    gender_dic,age_dic,doc_title_dic,users_dic,docs_dic = cPickle.load(open(path+'features.dic','r'))
    dics_tuple = [len(gender_dic),len(age_dic),len(doc_title_dic)]
    with codecs.open(path+dataset,'r') as train:
        for row in train.readlines():
            key, value = row.strip().split(':VS:')
            query = key.split('\t')
            response = value.split('\t')
            if response[-1] != '1.0':
                continue
            q = query[3].split('_')
            pos = response[4].split('_')
            #values = asker_map[key]
            #values.remove(value)
            #neg_index = random.randint(0,len(values)-1)
            neg_index = random.randint(0,len(responses)-1)
            neg_resp = responses[neg_index].split('\t')
            neg = neg_resp[4].split('_')

            q_profile = [0.0]*sum(dics_tuple[:2])
            pos_profile = [0.0]*dics_tuple[2]
            neg_profile = [0.0]*dics_tuple[2]
            q_profile[int(query[1])] = 1.0
            q_profile[int(query[2])+dics_tuple[0]] = 1.0
            
            pos_profile[int(response[1])] = 1.0
            pos_profile.append(response[2])
            pos_profile.append(response[3])
            
            neg_profile[int(neg_resp[1])] = 1.0
            neg_profile.append(neg_resp[2])
            neg_profile.append(neg_resp[3])

            embedding_samples.append([map(int,item) for item in [q,pos,neg]])
            profile_samples.append([map(float, item) for item in [q_profile, pos_profile, neg_profile]])
    res = np.array(embedding_samples),np.array(profile_samples) 
    cPickle.dump(res,open(path+'train_samples','w'))
    return res  #  must np.array or can't use [:,0] for list

def generate_test_samples():
    eb_samples, pro_samples, asker_label = [], [], []
    gender_dic,age_dic,doc_title_dic,users_dic,docs_dic = cPickle.load(open(path+'features.dic','r'))
    dics_tuple = [len(gender_dic),len(age_dic),len(doc_title_dic)]
    with codecs.open(path+'test_feature','r') as test:
        for row in test.readlines():
            key, value = row.strip().split(':VS:')
            query = key.split('\t')
            response = value.split('\t')
            q = map(int, query[3].split('_'))
            pos = map(int, response[4].split('_'))
            eb_samples.append([q,pos,pos])
            label = response[-1]
            asker_label.append((key,label))
            
            q_profile = [0.0]*sum(dics_tuple[:2])
            pos_profile = [0.0]*dics_tuple[2]
            q_profile[int(query[1])] = 1.0
            q_profile[int(query[2])+ dics_tuple[0]] = 1.0

            pos_profile[int(response[1])] = 1.0
            pos_profile.append(float(response[2]))
            pos_profile.append(float(response[3]))

            profile = [q_profile, pos_profile,pos_profile]
            pro_samples.append(profile)
    res = np.array(eb_samples), np.array(pro_samples), asker_label
    cPickle.dump(res,open(path+'test_samples','w'))
    return res

def generate_test_random_samples(batch_size=80):
    eb_samples, pro_samples, asker_label = [], [], []
    responses = cPickle.load(open(path+'responses','r'))
    gender_dic,age_dic,doc_title_dic,users_dic,docs_dic = cPickle.load(open(path+'features.dic','r'))
    dics_tuple = [len(gender_dic),len(age_dic),len(doc_title_dic)]
    with codecs.open(path+'test_feature','r') as test:
        for row in test.readlines():
            key, value = row.strip().split(':VS:')
            query = key.split('\t')
            response = value.split('\t')
            if response[-1] != '1.0':
                continue
            q = map(int, query[3].split('_'))
            q_profile = [0.0]*sum(dics_tuple[:2])
            q_profile[int(query[1])] = 1.0
            q_profile[int(query[2])+ dics_tuple[0]] = 1.0
            
            candidate_resp = [response]
            candidate_resp.extend(list(np.random.choice(responses,size=[batch_size-1],replace= False)))
            
            for i in range(batch_size):
                if i != 0:
                    response = candidate_resp[i].split('\t')
                    asker_label.append((key,'0.0'))
                else:
                    response = candidate_resp[i]
                    asker_label.append((key,'1.0'))
                pos = map(int, response[4].split('_'))

                pos_profile = [0.0]*dics_tuple[2]

                pos_profile[int(response[1])] = 1.0
                pos_profile.append(float(response[2]))
                pos_profile.append(float(response[3]))

                profile = [q_profile, pos_profile,pos_profile]
                pro_samples.append(profile)
                eb_samples.append([q,pos,pos])
    res = np.array(eb_samples), np.array(pro_samples), asker_label
    cPickle.dump(res,open(path+'test_samples','w'))
    return res


if __name__ == "__main__":
    extract_data_with_best_answer1('xywy.csv')
    #feature_process2('xywy.csv','raw_feature')
    #split_train_test3('raw_feature')
    #generate_uniform_pair('train_feature')
    #generate_test_samples()
    #generate_test_random_samples(10)
    print 'beginning...'

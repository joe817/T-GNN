from utils import *
from evaluator import *
from model import *
from sampling import *

import os
import pickle

import tensorflow as tf
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import math

if __name__ == '__main__':
    
    embed_dim=64
    metric_dim=32
    batch_size=512
    learning_rate=1e-3
    l2 = 1e-7
    neg_num = 3
    total_batch = 10000
    display_batch = 100
    
    data = load_data('gene','dblpdata.pkl')
    '''
    data = {}
    data['feature'] = {'P':p_emb, 'A':a_emb,'V':v_emb}
    data['HTS']=[['V','P','A'],['A','P','V']]
    data['adjs']=[[PV,AP],[PA,VP]]
    data['node_label'] = [p_label, a_label, v_label]
    '''

    p_label = data['node_label']['P']
    a_label = data['node_label']['A']
    v_label = data['node_label']['V']
    evaluator (data['feature'], p_label, a_label, v_label)


    model = TGNN(data, 'Perceptron', embed_dim, metric_dim, batch_size, learning_rate, l2)
    #model = TGNN(data, 'Bilinear', embed_dim, metric_dim, batch_size, learning_rate, l2)
    #model = TGNN(data, 'None', embed_dim, metric_dim, batch_size, learning_rate, 1e-3)
    
    u_s, u_d= sampling_paths(data, neg_num, numwalks=2, size=2)
    print (len(u_s), len(u_d))
    
    avg_loss = 0.
    for i in range(total_batch):
        
        sdx=(i*batch_size)%len(u_s)
        edx=((i+1)*batch_size)%len(u_s)
        if edx>sdx:  u_si = u_s[sdx:edx]
        else:  u_si = u_s[sdx:]+u_s[0:edx]
            
        sdx=(i*batch_size)%len(u_d)
        edx=((i+1)*batch_size)%len(u_d)
        if edx>sdx:  u_di = u_d[sdx:edx]
        else:  u_di = u_d[sdx:]+u_d[0:edx]

        loss= model.train_line(u_si, u_di)
        avg_loss += loss / display_batch
        if i % display_batch == 0 and i > 0:
            print ('%d/%d loss %8.6f' %(i,total_batch,avg_loss))
            avg_loss = 0.
            
            embed_dict = model.cal_embed()
            evaluator (embed_dict, p_label, a_label, v_label)
    
    
    embed_dict = model.cal_embed()
    with open(os.path.join('embed_dict.pkl'), 'wb') as file:
        pickle.dump(embed_dict, file)

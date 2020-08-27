from sklearn import metrics
import sklearn.preprocessing as preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
import numpy as np

def evaluator (embed_dict, p_label, a_label, v_label):
    embs =embed_dict['P']
    label = p_label
    
    estimator = KMeans(n_clusters=4)
    estimator.fit(embs)
    label_pred = estimator.labels_ 
    
    print ('NMI:%.4f' %metrics.normalized_mutual_info_score(label, label_pred))
    print ('ARI:%.4f' %metrics.adjusted_rand_score(label, label_pred))
    
    scaler=preprocessing.StandardScaler()
    X = scaler.fit_transform(embs)
    X = np.mat(X)
    train_X,test_X, train_y, test_y = train_test_split(X,label,test_size = 0.6)
    model = LogisticRegression(max_iter = 1000)
    model.fit(train_X, train_y)
    pred_y = model.predict(test_X)

    print('Micro-F1: %.4f' %f1_score(test_y, pred_y, average='micro'))
    print('Macro-F1: %.4f' %f1_score(test_y, pred_y, average='macro')) 
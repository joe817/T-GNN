from utils import *

class TGNN():
    def __init__(self, data, metric, embed_dim=64, metric_dim=32, batch_size=512, learning_rate=1e-3, l2 = 1e-7):
        
        tf.reset_default_graph()
        
        self.metricname= metric
        self.embed_dim = embed_dim 
        self.metric_dim = metric_dim
        self.sample_num = batch_size
        self.learning_rate = learning_rate
        self.l2 = l2

        self.node_num =0
        
        self.features = {} # initial embedding
        for nti in data['feature']:
            self.features[nti] = tf.convert_to_tensor(data['feature'][nti])
            self.node_num += len(data['feature'][nti])
        
        self.HTS = data['HTS'] # Hierarchical Tree Structures
        self.adjs = [] 
        for si in data['adjs']:
            adjs_si=[]
            for adj in si:
                adjs_si.append(cal_matrix(adj))
            self.adjs.append(adjs_si)
        
        self._construct_network()
        self._optimize_line()
        
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

      
    def _construct_network(self):
        
        nti_emb ={}

        for nti in self.features: #self aggregation
            nti_emb[nti] = []
            self_emb = tf.layers.dense(self.features[nti], self.embed_dim, use_bias=False)
            nti_emb[nti].append(self_emb)
            
        for htsi,adjs in zip(self.HTS,self.adjs): # hierarchical aggregation
            gruCell = tf.nn.rnn_cell.GRUCell(self.embed_dim, reuse = tf.AUTO_REUSE, name ="".join(htsi))   
            h_hat=[]
            h=[]
            for a, nti in enumerate(htsi):
                if a==0:
                    h_hat.append(self.features[nti])
                else:
                    h.append(tf.matmul(adjs[a-1], tf.layers.dense(h_hat[a-1],self.embed_dim, use_bias=False)))
                    output,state = gruCell(self.features[nti], h[a-1])
                    nti_emb[nti].append(output)
                    h_hat.append(state)
        #print (nti_emb)
        
        self.final_emb={} 
        for nti in nti_emb: # intergrating
            embs = nti_emb[nti]
            a1 = tf.layers.dense(embs[0],1,use_bias=False)
            embs = tf.convert_to_tensor(embs)
            a2 = tf.layers.dense(embs,1,use_bias=False)
            a = a2 + a1
            alpha = tf.nn.softmax(tf.nn.leaky_relu(a),dim=0)
            emb_i = tf.nn.relu(tf.reduce_sum(tf.multiply(alpha,embs),axis=0))
            self.final_emb[nti] = emb_i

        if self.metricname == 'Bilinear':
            self.embs = tf.concat([self.final_emb[nti] for nti in self.final_emb], axis = 0) 
            self.metrics = tf.Variable(xavier_init([4, self.embed_dim, self.embed_dim]))
            
        elif self.metricname == 'Perceptron':
            self.embs = tf.concat([tf.layers.dense(self.final_emb[nti],self.metric_dim,use_bias=False) 
                                   for nti in self.final_emb], axis = 0)   
            self.metrics = tf.Variable(xavier_init([4,self.metric_dim]))
            
        else:
            self.embs = tf.concat([self.final_emb[nti] for nti in self.final_emb], axis = 0)

                    

    def _optimize_line(self):
        """
        Unsupervised traininig
        """
        
        self.u_s = tf.placeholder(name='u_id', dtype=tf.int32, shape=[self.sample_num,3]) #node pair with same types
        self.u_d = tf.placeholder(name='u_id', dtype=tf.int32, shape=[self.sample_num,4]) #node pair with distinct types
        
        self.u_i_d = self.u_d[:,0]
        self.u_j_d = self.u_d[:,1]
        self.label_d = tf.cast(self.u_d[:,2],tf.float32)
        self.r = self.u_d[:,3]
        
        self.u_i_s = self.u_s[:,0]
        self.u_j_s = self.u_s[:,1]
        self.label_s = tf.cast(self.u_s[:,2],tf.float32)
        

        
        self.u_i_embedding_d = tf.matmul(tf.one_hot(self.u_i_d, depth=self.node_num, 
                                                  dtype=tf.float32), self.embs)
        self.u_j_embedding_d = tf.matmul(tf.one_hot(self.u_j_d, depth=self.node_num, 
                                                  dtype=tf.float32), self.embs)
        self.u_i_embedding_s = tf.matmul(tf.one_hot(self.u_i_s, depth=self.node_num, 
                                                  dtype=tf.float32), self.embs)
        self.u_j_embedding_s = tf.matmul(tf.one_hot(self.u_j_s, depth=self.node_num, 
                                                  dtype=tf.float32), self.embs)

        #Relational Metric learning
        if self.metricname == 'Bilinear':
            M_r = tf.nn.embedding_lookup(self.metrics,self.r)
            self.inner_product_d = tf.reduce_sum(tf.squeeze(tf.matmul(tf.expand_dims(self.u_i_embedding_d, 1),M_r),1) *  
                                               self.u_j_embedding_d, axis=1)
            
        elif self.metricname == 'Perceptron': 
            M_r = tf.nn.embedding_lookup(self.metrics,self.r)
            self.inner_product_d = tf.reduce_sum(M_r * tf.nn.tanh(self.u_i_embedding_d + self.u_j_embedding_d), axis=1)
            
        else:
            self.inner_product_d = tf.reduce_sum(self.u_i_embedding_d * self.u_j_embedding_d, axis=1)
            
        self.inner_product_s = tf.reduce_sum(self.u_i_embedding_s * self.u_j_embedding_s, axis=1)

        self.loss = -tf.reduce_mean(tf.log_sigmoid(self.label_d * self.inner_product_d)
                                   )-tf.reduce_mean(tf.log_sigmoid(self.label_s * self.inner_product_s))
                                                  
        self.l2_loss = self.l2 * sum(tf.nn.l2_loss(var) #l2 norm
            for var in tf.trainable_variables() if 'bias' not in var.name)
        self.loss = self.loss + self.l2_loss
        self.line_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        
    def train_line(self,u_s, u_d):
        """
        Train one minibatch.
        """
        feed_dict = {self.u_s: u_s, self.u_d: u_d}
        _, loss = self.sess.run((self.line_optimizer, self.loss), feed_dict=feed_dict)
        return loss
    
    def cal_embed(self):
        return self.sess.run(self.final_emb)
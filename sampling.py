from utils import *
class AliasSampling:
    def __init__(self, prob):
        self.n = len(prob)
        self.U = np.array(prob) * self.n
        self.K = [i for i in range(len(prob))]
        overfull, underfull = [], []
        for i, U_i in enumerate(self.U):
            if U_i > 1:
                overfull.append(i)
            elif U_i < 1:
                underfull.append(i)
        while len(overfull) and len(underfull):
            i, j = overfull.pop(), underfull.pop()
            self.K[j] = i
            self.U[i] = self.U[i] - (1 - self.U[j])
            if self.U[i] > 1:
                overfull.append(i)
            elif self.U[i] < 1:
                underfull.append(i)

    def sampling(self, n=1):
        x = np.random.rand(n)
        i = np.floor(self.n * x)
        y = self.n * x - i
        i = i.astype(np.int32)
        res = [i[k] if y[k] < self.U[i[k]] else self.K[i[k]] for k in range(n)]
        if n == 1:
            return res[0]
        else:
            return res

def sampling_paths(data, negative_num, numwalks=2, size= 2):
    #data['adjs']=[[PV,AP],[PA,VP]]
    ap_adj = data['adjs'][0][1]
    pa_adj = ap_adj.T
    vp_adj = data['adjs'][1][1]
    pv_adj = vp_adj.T
    
    p_num = len(pa_adj)
    a_num = len(ap_adj)
    v_num = len(vp_adj)
    
    if os.access("gene/all_negative_samplings", os.F_OK):
        all_negative_samplings = load_data('gene', 'all_negative_samplings')
        all_neighbor_samplings = load_data('gene', 'all_neighbor_samplings')
    else:
        all_neighbor_samplings=[]
        all_negative_samplings=[]
        for i, adj in enumerate([ap_adj,pv_adj,vp_adj,pa_adj]):
            samplings = []
            n_samplings = []
            for j in range(len(adj)):
                node_weights = adj[j]
                weight_distribution = node_weights / np.sum(node_weights)            
                samplings.append(AliasSampling(weight_distribution))
                n_weight_distribution = (node_weights-1) / np.sum((node_weights-1)) 
                n_samplings.append(AliasSampling(n_weight_distribution))
            all_neighbor_samplings.append(samplings)
            all_negative_samplings.append(n_samplings)
        dump_data(all_neighbor_samplings, 'gene', 'all_neighbor_samplings')
        dump_data(all_negative_samplings, 'gene', 'all_negative_samplings')

    
    u_d = [] #distinct type [[u_i, u_j, label, r]], r:relation type
    u_s = [] #same type [[u_i, u_j, label]]

    for i in range(numwalks):
        for p in range(p_num):
            if 1<=size:
                # P-V
                v = all_neighbor_samplings[1][p].sampling()
                u_d.append([p, v+a_num+p_num, 1, 0])
                for k in range(negative_num):
                    v_n = all_negative_samplings[1][p].sampling()
                    u_d.append([p, v_n+a_num+p_num, -1, 0])

                # P-A                    
                a = all_neighbor_samplings[3][p].sampling()
                u_d.append([p, a+p_num, 1, 1])
                for k in range(negative_num):
                    a_n = all_negative_samplings[3][p].sampling()
                    u_d.append([p, a_n+p_num, -1, 1])
                    
            if 2<=size:    
                # P-V-P    
                p1 = all_neighbor_samplings[2][v].sampling()
                u_s.append([p, p1, 1])
                for k in range(negative_num):
                    p_n = all_negative_samplings[2][v].sampling()
                    u_s.append([p, p_n, -1])

                # P-A-P 
                p1 = all_neighbor_samplings[0][a].sampling()
                u_s.append([p, p1, 1])
                for k in range(negative_num):
                    p_n = all_negative_samplings[0][a].sampling()
                    u_s.append([p, p_n, -1])

                # A-P-V
                v = all_neighbor_samplings[1][p1].sampling()
                u_d.append([a+p_num, v+a_num+p_num, 1, 2])
                for k in range(negative_num):
                    v_n = all_negative_samplings[1][p1].sampling()
                    u_d.append([a+p_num, v_n+a_num+p_num, -1, 2])

                # A-P-A
                a1 = all_neighbor_samplings[3][p1].sampling()
                u_s.append([a+p_num, a1+p_num, 1])
                for k in range(negative_num):
                    a_n = all_negative_samplings[3][p1].sampling()
                    u_s.append([a+p_num, a_n+p_num, 1])

    return u_s, u_d
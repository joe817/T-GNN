# T-GNN

![](https://github.com/joe817/img/blob/master/TGNN.png)

## Basic requirements

* python 3.7.7
* tensorflow 1.15.0
* numpy 1.18.5

## Data description

You may need to prepare the data in the following format：

```bash
#dict
data = {} 
#the initial emebdding of nodes, the key is node type.
data['feature'] = {'P':p_emb, 'A':a_emb,'V':v_emb} 
#Hierarchical tree structures(HTS), i.e., VPA, APV.
data['HTS']=[['V','P','A'],['A','P','V']]
#The adjacency matrix between each two levels in each hierarchical tree
data['adjs']=[[PV,AP],[PA,VP]]
'''

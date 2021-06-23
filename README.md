# T-GNN

![](https://github.com/joe817/img/blob/master/TGNN.png)

@INPROCEEDINGS{9338420,
  author={Qiao, Ziyue and Wang, Pengyang and Fu, Yanjie and Du, Yi and Wang, Pengfei and Zhou, Yuanchun},
  booktitle={2020 IEEE International Conference on Data Mining (ICDM)}, 
  title={Tree Structure-Aware Graph Representation Learning via Integrated Hierarchical Aggregation and Relational Metric Learning}, 
  year={2020},
  volume={},
  number={},
  pages={432-441},
  doi={10.1109/ICDM50108.2020.00052}}

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
```


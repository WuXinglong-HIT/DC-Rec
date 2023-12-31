import datetime
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.nn import Module, Parameter
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class CombineGraph(Module):
    def __init__(self, opt, num_node, adj_all, num,  hiddenSize, embedding_dim,preemb,preemb1, n_layers = 1):
        super(CombineGraph, self).__init__()
        self.opt = opt
        self.n_items = num_node
        self.hidden_size = hiddenSize
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.batch_size = opt.batch_size
        self.num_node = num_node
        self.dim = opt.hiddenSize
        self.dropout_local = opt.dropout_local
        self.dropout_global = opt.dropout_global
        self.dropout=0
        self.hop = opt.n_iter
        self.w_3=nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.sample_num = opt.n_sample
        self.adj_all = trans_to_cuda(torch.Tensor(adj_all)).long()
        self.num = trans_to_cuda(torch.Tensor(num)).float()
        self.emb = nn.Embedding(self.n_items, self.embedding_dim, padding_idx = 0)#embeding come from
        self.emb.weight=torch.nn.Parameter(preemb1)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.n_layers)
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)
        self.ct_dropout = nn.Dropout(0.8)
        self.b = nn.Linear(self.embedding_dim, 2 * self.hidden_size, bias=False)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_agg = []
        self.embedding = nn.Embedding(num_node, self.dim)
        self.embedding.weight=torch.nn.Parameter(preemb)#####r
        self.pos_embedding = nn.Embedding(300, self.dim)
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.glu1 = nn.Linear(self.dim, self.dim)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)
        self.linear_transform = nn.Linear(self.dim, self.dim, bias=False)

        self.leakyrelu = nn.LeakyReLU(opt.alpha)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.wg=nn.Parameter(torch.Tensor( self.dim, self.dim))
        self.ws=nn.Parameter(torch.Tensor( self.dim, self.dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def sample(self, target, n_sample):
        # neighbor = self.adj_all[target.view(-1)]
        # index = np.arange(neighbor.shape[1])
        # np.random.shuffle(index)
        # index = index[:n_sample]
        # return self.adj_all[target.view(-1)][:, index], self.num[target.view(-1)][:, index]
        return self.adj_all[target.view(-1)], self.num[target.view(-1)]
    def init_hidden(self, batch_size):
        return torch.zeros((self.n_layers, batch_size, self.hidden_size), requires_grad=True).to(self.device)
    def compute_scores(self, hidden, mask):
        mask = mask.float().unsqueeze(-1)

        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        kkk=hs
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)#nh==zi
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * hidden, 1)#select就是要找的被监督向量
        b = self.embedding.weight[1:]  # n_nodes-1 x latent_size
        scores = torch.matmul(select, b.transpose(1, 0))
        return scores,select

    def SSL(self, sess_emb_hgnn, sess_emb_lgcn):
        def row_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            return corrupted_embedding
        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:,torch.randperm(corrupted_embedding.size()[1])]
            return corrupted_embedding
        def score(x1, x2):
            return torch.matmul(torch.matmul(x1,self.w_3 ),x2)
        abc=0
        cba=0
        qqq=[]
        pos = score(sess_emb_hgnn, sess_emb_lgcn.transpose(0,1))
        neg1 = score(sess_emb_hgnn, sess_emb_hgnn.transpose(0,1))
        e=torch.eye(pos.shape[0])
        e = trans_to_cuda(e).float()
        masked=torch.mul(pos,e)
        posed=sum(masked,1).view(1,pos.shape[0])
        neg1ed=torch.cat((neg1,posed),0)
        kkk=torch.max(neg1ed,0)[0].view(1,pos.shape[0])
        real_pos=torch.exp(posed-kkk)#pos[i][i]
        neg2=torch.exp(torch.sum(torch.mul(neg1,e),1).view(1,pos.shape[0])-kkk)#neg[i][i]
        neged=torch.sum(torch.exp(neg1-kkk),0).view(1,pos.shape[0])#neg[i][j]
        con_loss=-torch.sum(real_pos/(real_pos+neged-neg2+1e-8))
        return con_loss

    def forward(self, inputs, adj, mask_item, item, seq, lengths,order):
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]
        h = self.embedding(inputs)
        A=adj
        batch_size1 = h.shape[0]
        N=h.shape[1]
        A1=torch.sum(A,1).view(batch_size1,N,1)
        A2=torch.sum(A,2).view(batch_size1,1,N)
        A3=torch.matmul(A1,A2)
        A3[torch.where(A3==0)]=-1
        A3=1/A3
        A3[torch.where(A3==-1)]=0
        A3=torch.sqrt(A3)
        A=torch.mul(A3,A)
        h_local= torch.matmul(A, h)
        
        
        
        item_neighbors = [inputs]
        weight_neighbors = []
        support_size = seqs_len

        for i in range(1, self.hop + 1):
            item_sample_i, weight_sample_i = self.sample(item_neighbors[-1], self.sample_num)
            support_size *= self.sample_num
            item_neighbors.append(item_sample_i.view(batch_size, support_size))
            weight_neighbors.append(weight_sample_i.view(batch_size, support_size))

        entity_vectors = [self.embedding(i) for i in item_neighbors]
        weight_vectors = weight_neighbors

        session_info = []
        item_emb = self.embedding(item) * mask_item.float().unsqueeze(-1)
        neighbor_vector=[]
        # mean 
        sum_item_emb = torch.sum(item_emb, 1) / torch.sum(mask_item.float(), -1).unsqueeze(-1)
        
        # sum
        # sum_item_emb = torch.sum(item_emb, 1)
        
        sum_item_emb = sum_item_emb.unsqueeze(-2)
        for i in range(self.hop):
            session_info.append(sum_item_emb.repeat(1, entity_vectors[i].shape[1], 1))

        for n_hop in range(self.hop):
            entity_vectors_next_iter = []
            shape = [batch_size, -1, self.sample_num, self.dim]
            for hop in range(self.hop - n_hop):
            
                if session_info[hop] is not None:
                    alpha = torch.softmax(weight_vectors[hop].view(batch_size, -1, self.sample_num), -1).unsqueeze(-1)
                    neighbor_vector = torch.sum(alpha * entity_vectors[hop+1].view(shape), dim=-2)
                else:
                    neighbor_vector = torch.mean(entity_vectors[hop+1].view(shape), dim=2)
                output1 =  neighbor_vector
                output1 = F.dropout(output1, self.dropout, training=self.training)
                output1 = output1.view(batch_size, -1, self.dim)
                vector = torch.relu(output1)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        h_global = entity_vectors[0].view(batch_size, seqs_len, self.dim)

        h_local = F.dropout(h_local, self.dropout_local, training=True)
        h_global = F.dropout(h_global, self.dropout_global, training=True)
        resetgate=torch.sigmoid(torch.matmul(h_local,self.ws) + torch.matmul(h_global,self.wg))
        output=h_global*resetgate+h_local-h_local*resetgate
        hidden = self.init_hidden(seq.size(1))
        embs=F.dropout(self.emb(seq), self.dropout_local, training=True)
        embs = pack_padded_sequence(embs, lengths)
        gru_out, hidden = self.gru(embs, hidden)
        gru_out, lengths = pad_packed_sequence(gru_out)
        ht = hidden[-1]
        gru_out = gru_out.permute(1, 0, 2)
        c_global = ht
        q1 = self.a_1(gru_out.contiguous().view(-1, self.hidden_size)).view(gru_out.size())  
        q2 = self.a_2(ht)
        mask = torch.where(seq.permute(1, 0) > 0, torch.tensor([1.], device = self.device), torch.tensor([0.], device = self.device))
        q2_expand = q2.unsqueeze(1).expand_as(q1)
        q2_masked = mask.unsqueeze(2).expand_as(q1) * q2_expand
        alpha = self.v_t(torch.sigmoid(q1 + q2_masked).view(-1, self.hidden_size)).view(mask.size())
        c_local = torch.sum(alpha.unsqueeze(2).expand_as(gru_out) * gru_out, 1)
        c_t = torch.cat([c_local, c_global], 1)
        b_t= c_local
        tiao=np.arange(len(order))
        for i in np.arange(len(order)):
            if i==0:
                continue
            if order[i]<order[i-1]:
               j=i
               while True:
                 k=order[j]
                 order[j]=order[j-1]
                 order[j-1]=k
                 kkk=tiao[j]
                 tiao[j]=tiao[j-1]
                 tiao[j-1]=kkk
                 j=j-1
                 if j==0 or order[j]>order[j-1]:
                     break
        b_t=b_t[tiao]
        return output,b_t


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

     
def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, data):
    alias_inputs, adj, items, mask, targets, inputs = data
    seq=inputs
    alias_inputs = trans_to_cuda(alias_inputs).long()
    items = trans_to_cuda(items).long()
    adj = trans_to_cuda(adj).float()
    mask = trans_to_cuda(mask).long()
    inputs = trans_to_cuda(inputs).long()
    lengths=[]
    shu=0
    for i in seq:
        for j in i:
            if j!=0:
                shu=shu+1
        lengths.append(shu)
        shu=0
    order=np.arange(len(lengths))
    for i in np.arange(len(lengths)):
            if i==0:
                continue
            if lengths[i]>lengths[i-1]:
               j=i
               while True:
                 k=lengths[j]
                 lengths[j]=lengths[j-1]
                 lengths[j-1]=k
                 kkk=order[j]
                 order[j]=order[j-1]
                 order[j-1]=kkk
                 j=j-1
                 if j==0 or lengths[j]<=lengths[j-1]:
                     break
    seq=seq[order]

    seq=seq[:,:lengths[0]]
    seq=seq.transpose(0,1).to(device)
    hidden,b_t = model(items, adj, mask, inputs,seq,lengths,order)
    get = lambda index: hidden[index][alias_inputs[index]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])    
    scores,select=model.compute_scores(seq_hidden, mask)
    scores = trans_to_cuda(scores).float()
    b_t = trans_to_cuda(b_t).float()
    b = model.emb.weight[1:]  # n_nodes-1 x latent_size
    scores2 = torch.matmul(b_t, b.transpose(1, 0))
    
    con_loss = model.SSL(select, b_t)
    con_loss =con_loss+model.SSL(b_t,select)
    scores3 = torch.matmul(b_t+select, b.transpose(1, 0)+model.embedding.weight[1:].transpose(1, 0))
    return targets,scores,0.001*con_loss,scores2 ,scores3 #beta

def getDCG(scores):
    return np.sum(
        np.divide(np.power(2, scores) - 1, np.log2(np.arange(scores.shape[0], dtype=np.float32) + 2)),
        dtype=np.float32)

def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=model.batch_size,
                                               shuffle=True, pin_memory=True)

    for data in tqdm(train_loader):
    #    print(data)
    #    print(len(data))
     #   print(data[0])
     
        model.optimizer.zero_grad()

        targets, scores,con_loss,scores2,scores3 = forward(model, data)
        targets = trans_to_cuda(targets).long()
        loss = model.loss_function(scores, targets - 1)+con_loss+model.loss_function(scores2, targets - 1)
        #loss = con_loss
        loss.backward()
        model.optimizer.step()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True)
    result = []
    hit5, mrr5,ndcg5,precision5 = [], [],[],[]
    hit10, mrr10,ndcg10,precision10 = [], [],[],[]
    hit20, mrr20,ndcg20,precision20 = [], [],[],[]
    for data in test_loader:
        targets, scores,con_loss,scores2,scores3= forward(model, data)
        sub_scores5 = scores.topk(5)[1]
        sub_scores10 = scores.topk(10)[1]
        sub_scores20 = scores.topk(20)[1]
        sub_scores5 = trans_to_cpu(sub_scores5).detach().numpy()
        sub_scores10 = trans_to_cpu(sub_scores10).detach().numpy()
        sub_scores20 = trans_to_cpu(sub_scores20).detach().numpy()
        targets = targets.numpy()
        for score, target, mask in zip(sub_scores5, targets, test_data.mask):
            hit5.append(np.isin(target - 1, score))
            bbc=score
            if len(np.where(score == target - 1)[0]) == 0:
                mrr5.append(0)
                precision5.append(0)
            else:
                mrr5.append(1 / (np.where(score == target - 1)[0][0] + 1))
                precision5.append(0.05)
            if np.isin(target - 1, score)==True:
                for i in torch.arange(len(score)):
                    if bbc[i]==target-1:
                        bbc[i]=1
                    else :
                        bbc[i]=0
             
                ndcg5.append(getDCG(bbc))
                 
            else:
                ndcg5.append(0)    
        for score, target, mask in zip(sub_scores10, targets, test_data.mask):
            hit10.append(np.isin(target - 1, score))
            bbc=score
            if len(np.where(score == target - 1)[0]) == 0:
                mrr10.append(0)
                precision10.append(0)
            else:
                mrr10.append(1 / (np.where(score == target - 1)[0][0] + 1))
                precision10.append(0.05)
            if np.isin(target - 1, score)==True:
                for i in torch.arange(len(score)):
                    if bbc[i]==target-1:
                        bbc[i]=1
                    else :
                        bbc[i]=0
             
                ndcg10.append(getDCG(bbc))
                 
            else:
                ndcg10.append(0)    
        for score, target, mask in zip(sub_scores20, targets, test_data.mask):
            hit20.append(np.isin(target - 1, score))
            bbc=score
            if len(np.where(score == target - 1)[0]) == 0:
                mrr20.append(0)
                precision20.append(0)
            else:
                mrr20.append(1 / (np.where(score == target - 1)[0][0] + 1))
                precision20.append(0.05)
            if np.isin(target - 1, score)==True:
                for i in torch.arange(len(score)):
                    if bbc[i]==target-1:
                        bbc[i]=1
                    else :
                        bbc[i]=0
             
                ndcg20.append(getDCG(bbc))
                 
            else:
                ndcg20.append(0)    

    result.append(np.mean(hit20) * 100)
    result.append(np.mean(mrr20) * 100)
    hit5=np.mean(hit5) * 100
    mrr5=np.mean(mrr5) * 100
    ndcg5= np.mean(ndcg5) * 100
    precision5=np.mean(precision5)*100
    hit10=np.mean(hit10) * 100
    mrr10=np.mean(mrr10) * 100
    ndcg10= np.mean(ndcg10) * 100
    precision10=np.mean(precision10)*100
    hit20=np.mean(hit20) * 100
    mrr20=np.mean(mrr20) * 100
    ndcg20= np.mean(ndcg20) * 100
    precision20=np.mean(precision20)*100
    print('hit:')
    print(hit5)
    print(hit10)
    print(hit20)
    print('mrr:')
    print(mrr5)
    print(mrr10)
    print(mrr20)
    print('precision:')
    print(precision5)
    print(precision10)
    print(precision20)
    print('ndcg:')
    print(ndcg5)
    print(ndcg10)
    print(ndcg20)
    return result

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy


class Aggregator(nn.Module):
    def __init__(self, batch_size, dim, dropout, act, name=None):
        super(Aggregator, self).__init__()
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.dim = dim

    def forward(self):
        pass


class LocalAggregator(nn.Module):
    def __init__(self, dim, alpha, dropout=0., name=None):
        super(LocalAggregator, self).__init__()
        self.dim = dim
        self.dropout = dropout

        self.a_0 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_1 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_3 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.bias = nn.Parameter(torch.Tensor(self.dim))
        self.hidden_size = 100
        self.input_size = 100 * 2
        self.gate_size = 3 * 100
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        #self.w_ih = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.w_kk = Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, hidden, A, mask_item=None):
       # h = hidden
        batch_size = hidden.shape[0]
        #N = h.shape[1]      
        N=hidden.shape[1]

        A1=torch.sum(A,1).view(batch_size,N,1)
        A2=torch.sum(A,2).view(batch_size,1,N)
        A3=torch.matmul(A1,A2)
        A3[torch.where(A3==0)]=-1
        A3=1/A3
        A3[torch.where(A3==-1)]=0
        A3=torch.sqrt(A3)
        A=torch.mul(A3,A)
        hy= torch.matmul(A, hidden)
        #input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah

       # input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah

       # inputs = torch.cat([input_in, input_out], 2)
       ## gi = F.linear(inputs, self.w_ih, self.b_ih)
       # gh = F.linear(hidden, self.w_hh, self.b_hh)
       # i_r, i_i, i_n = gi.chunk(3, 2)
       # h_r, h_i, h_n = gh.chunk(3, 2)

      #  resetgate = torch.sigmoid(i_r + h_r)
      #  inputgate = torch.sigmoid(i_i + h_i)
       # newgate = torch.tanh(i_n + resetgate * h_n)
       # hy = newgate + inputgate * (hidden - newgate)
      #  gi = F.linear(inputs, self.w_ih, self.b_ih)
       # gh = F.linear(hidden, self.w_hh, self.b_hh)
       # i_r, i_i, i_n,i_o = gi.chunk(4, 2)
       # h_r, h_i, h_n,h_o = gh.chunk(4, 2)
       # i_t = torch.sigmoid(i_r + h_r)
       # f_t = torch.sigmoid(i_i + h_i)
       # g_t = torch.tanh(i_n+h_n)
       # o_t = torch.sigmoid(i_o+h_o)
        # 记忆单元和隐藏单元更新
        #c_t = f_t * hidden + i_t * g_t
       # h_t = o_t * torch.tanh(c_t)
        #resetgate = torch.sigmoid(i_r + h_r)
        #inputgate = torch.sigmoid(i_i + h_i)
       # newgate = torch.tanh(i_n + resetgate * h_n)
       # hy = newgate + inputgate * (hidden - newgate)
        return hy



class GlobalAggregator(nn.Module):
    def __init__(self, dim, dropout, act=torch.relu, name=None):
        super(GlobalAggregator, self).__init__()
        self.dropout = dropout
        self.act = act
        self.dim = dim

        #self.w_1 = nn.Parameter(torch.Tensor(self.dim + 1, self.dim))
       # self.w_1 = torch.Tensor([[1]])
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.w_3 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.bias = nn.Parameter(torch.Tensor(self.dim))

    def forward(self, self_vectors, neighbor_vector, batch_size, masks, neighbor_weight, extra_vector=None):
        if extra_vector is not None:
            alpha = torch.softmax(neighbor_weight, -1).unsqueeze(-1)
            neighbor_vector = torch.sum(alpha * neighbor_vector, dim=-2)
        else:
            neighbor_vector = torch.mean(neighbor_vector, dim=2)
       

        output =  neighbor_vector
        output = F.dropout(output, self.dropout, training=self.training)
 
        output = output.view(batch_size, -1, self.dim)
        output = self.act(output)
        return output

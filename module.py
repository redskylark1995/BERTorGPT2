import torch, os
from torch import nn, optim
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from torch.nn import init, functional as F
from torch import nn
import random
import time

import config as cfg


class Attention(nn.Module):

    def __init__(self, isMask=False):
        super().__init__()
        self.dk = (cfg.embed_dim // cfg.head_num) ** 0.5
        self.isMask = isMask

        self.c_attn = nn.Linear(cfg.embed_dim, cfg.embed_dim * 3)

        self.attn_drop = nn.Dropout(0.1)
        self.resi_drop = nn.Dropout(0.1)

        self.c_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)

        if self.isMask:
            self.register_buffer("mask", torch.tril(torch.ones(cfg.pos_num, cfg.pos_num)))

    def forward(self, x):
        x = self.c_attn(x)
        x = x.reshape(*x.shape[:-1], cfg.head_num, -1)
        x = x.transpose(-2, -3)
        q, k, v = x.chunk(3, dim=-1)
        w = (q @ k.transpose(-1, -2)) / self.dk
        if self.isMask:
            mask = self.mask[0:w.size(-2), 0:w.size(-1)]
            w = w * mask - (1 - mask) * 1e5
        w = torch.softmax(w, dim=-1)
        w = self.attn_drop(w)

        a = w @ v

        a = a.transpose(-2, -3)
        a = a.reshape(*a.shape[:-2], cfg.embed_dim)

        h = self.c_proj(a)
        h = self.resi_drop(h)

        return h


class Block(nn.Module):

    def __init__(self, isMask=False):
        super().__init__()

        self.layer_normal_1 = nn.LayerNorm(cfg.embed_dim)

        self.attention = Attention(isMask)

        self.layer_normal_2 = nn.LayerNorm(cfg.embed_dim)

        self.proj = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.multi * cfg.embed_dim),
            nn.LeakyReLU(),
            nn.Linear(cfg.multi * cfg.embed_dim, cfg.embed_dim),
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        h = self.layer_normal_1(x)
        a = self.attention(h)
        a = a + x
        a = self.layer_normal_2(a)
        h = self.proj(a)
        h = self.dropout(h)
        y = h + a
        return y


class GPT2(nn.Module):

    def __init__(self,isMask=False):
        super().__init__()

        self.vocab_embed = nn.Embedding(cfg.vocab_num, cfg.embed_dim)
        a =self.vocab_embed(torch.tensor([[i for i in range(1,cfg.vocab_num)]]))[0]#词向量
        b = torch.zeros([1,cfg.embed_dim])
        c = torch.cat([b,a],dim=0)
        self.vocab_embed.weight.data.copy_(c)

        self.pos_embed = nn.Embedding(cfg.pos_num, cfg.embed_dim)#位置向量
        a =self.pos_embed(torch.tensor([[i for i in range(1,cfg.pos_num)]]))[0]#将位置0的向量设置为0
        b = torch.zeros([1,cfg.embed_dim])
        c = torch.cat([b,a],dim=0)
        self.pos_embed.weight.data.copy_(c)
        # self.type_embed = nn.Embedding(cfg.type_num, cfg.embed_dim)#类型向量

        self.blocks = []
        for _ in range(cfg.block_num):
            self.blocks.append(Block(isMask=isMask))

        self.drop = nn.Dropout(0.1)

        self.sequential = nn.Sequential(*self.blocks)
        linears = []
        for  i in range(cfg.linenum):
            
            linears.append(nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False))
            linears.append(nn.LeakyReLU())
            
        self.linear = nn.Sequential(*linears)
        
        self.lstm = nn.LSTM(cfg.embed_dim,cfg.embed_dim,num_layers=cfg.lstmnum,batch_first=True)
        self.output_layer = nn.Linear(cfg.embed_dim, 2, bias=False)
        # self.Softmax = torch.nn.Softmax()
    def rmnullword(self):
        a =self.pos_embed(torch.tensor([[i for i in range(1,cfg.pos_num)]]).to(torch.device(cfg.device)))[0]#将位置0的向量设置为0
        b = torch.zeros([1,cfg.embed_dim]).to(torch.device(cfg.device))
        c = torch.cat([b,a],dim=0)
        self.pos_embed.weight.data.copy_(c)
        a =self.vocab_embed(torch.tensor([[i for i in range(1,cfg.vocab_num)]]).to(torch.device(cfg.device)))[0]#词向量
        b = torch.zeros([1,cfg.embed_dim]).to(torch.device(cfg.device))
        c = torch.cat([b,a],dim=0)
        self.vocab_embed.weight.data.copy_(c)
    def forward(self, x, p,le):
        e = self.vocab_embed(x)
        p = self.pos_embed(p)
        h = self.drop(e + p )
        h = self.sequential(h)
        b = self.output_layer(h)
        h=self.linear(h)
        #生成lstm初始状态
        h0 = torch.zeros(cfg.lstmnum ,h.size(0), cfg.embed_dim).to(torch.device(cfg.device))
        c0 = torch.zeros(cfg.lstmnum,h.size(0), cfg.embed_dim).to(torch.device(cfg.device))

        out,(hn,cn)= self.lstm(h,(h0,c0)) 

        out = self.output_layer(out[[i for i in range(out.size(0))],le,:])
        return out


if __name__ == '__main__':
    
    gpt = GPT2()#torch.load(os.path.join("weights", "bert.pth"))
    gpt.to(torch.device(cfg.device))
    gpt.eval()
    
    os = []
    from dataset import *
    tedataset = MyDataset("逆天邪神正样本token.txt","逆天邪神负样本token.txt")
    #x = torch.tensor([[0,3,3,3,372,417,1894,1278,3,238,1052,247,3,1304,3,247,1578,1766,217,3,1949,3,1017,1052,247,3,1282,1337,1229,1018,3,379,705,1170,1786]]).cuda()
    x = torch.tensor([[0,3,3,3,1852,1052,187,3,1019,597,839,1275,3,1166,247,1,3,3,3,3,1042,476,1600,153,360,1835,900,153,1259,1835,72,1189,772,72,1374,1848,73,1287,73,1897,903,360,1835,900,1017,1745,66,1892,430]]).cuda()

    p = torch.tensor([[i for i  in range(x.shape[1])]]).cuda()
    for i in range(1):
        y = gpt(x, p)
        # y = y[:, -1:]
        # v, y = torch.topk(y, 2, dim=-1)

        # v, y = v.reshape(-1, 2), y.reshape(-1, 2)
        # v = torch.multinomial(torch.softmax(v, dim=-1), 1)
        # y = torch.gather(y, -1, v)

        # x = torch.cat([x, y], dim=1)
        # p = torch.tensor([range(i + 2)]).cuda()

        print(y)

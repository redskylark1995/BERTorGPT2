import os

line1 = open(r"词表.txt","r+",encoding="utf-8").read()
vocab_lenth = len(line1.split())#3423
print(vocab_lenth)
block_num = 6#Block层数
head_num = 12#注意力头数
embed_dim = 768#词向量长度
vocab_num = vocab_lenth#词表长度
pos_num = 500#位置编码长度
device = "cuda:0"
multi = 4#block全连接运算数
textclass =2#生成时与词表长度相同
stride = 1 #数据移动的长度
linenum=4
lstmnum=2
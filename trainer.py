import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from module import *
from dataset import *

def weight_init(m):
    if isinstance(m, nn.Linear):#类型相同，都是nn的类方法？？
        nn.init.xavier_normal_(m.weight)#初始化权重
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)#将m的偏置固定为常数，这里设为0

class Trainer:
    
    def __init__(self):
        self.net = GPT2(isMask=True)
        self.net.train()

        # self.weight_file_bak = os.path.join("weights", "apt2_k_bak.pt")torch.load(os.path.join("weights", "bert.pth"))
        self.weight_file = os.path.join("weights", "bert.pth")

        if not os.path.exists("weights"):
            os.makedirs("weights")

        if os.path.exists(self.weight_file):
            # self.net=torch.load(r"weights\bertminjq.pth")
            self.net=torch.load(self.weight_file)

            print("加载参数")
        # else:
        #     self.net.apply(weight_init)#如果参数路径不存在，就初始化w和b，用上面的函数

        # self.net = nn.DataParallel(self.gpt2, device_ids=[0, 2, 3])#多张显卡进行训练，不需要这个
        # self.net = self.gpt2
        self.net= self.net.to(torch.device(cfg.device))

        self.opt = optim.Adam(self.net.parameters(), lr=0.0001)

    def train(self):
        myDataset = MyDataset("逆天邪神token.txt")

        print(len(myDataset))
        dataloader = DataLoader(myDataset, batch_size=5, shuffle=True)
        device =torch.device(torch.device(cfg.device))
        epoch = 1
        loss_cros_fun = nn.CrossEntropyLoss()
        while True:
            # if epoch>500:
            #     break
            sum_loss = 0
            sum_jq = 0.0
            max_jq = 0.0
            for i, (x,p,label,le) in enumerate(dataloader):
               
                x,p, label = x.to(device),p.to(device),label.to(device)
                _y = self.net(x, p,le)#只传了x和p，没有要句编码
                _y =_y.reshape(-1, cfg.textclass)
                #之前网络跑不通是因为编码字典一共N个字，所以说词向量也是0到N-1才对（取的编码字典的索引），之前有8000,9000这种，早已超出标签

                label = label.reshape(-1)#相乘起来变成一个维度（N*500）
                loss = loss_cros_fun(_y, label)#交叉熵,(N,500,4413)he (N,500，1)进行对比，这么理解
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                self.net.rmnullword()
                # 
                jq = torch.sum(torch.argmax(_y,dim=1)==label)/5.0
                print("轮次：{}\t批次：{}\t损失：{:.4f}\t数据量：{}\t精确度{:.4f}".format(epoch, i, loss.cpu().detach().item(),label.size(0),jq))
                
                sum_loss += loss.cpu().detach().item()
                sum_jq += jq
                if i % 500 == 0 and i > 0:
                    torch.save(self.net, self.weight_file)
                    print("保存成功")
                    
                print("轮次：{}\t平均损失：{:.4f}\t平均精确度:{:.4f}".format(epoch, sum_loss /(i+1),sum_jq/(i+1)))
            # if epoch%10==0:
            torch.save(self.net, self.weight_file)
            if max_jq<(sum_jq/(i+1)):
                max_jq = sum_jq/(i+1)
                torch.save(self.net, "weights\\bertminjq.pth")
                print("更新最小精度")
            print("保存成功")
            epoch+=1
            print("完成一轮")


if __name__ == '__main__':
	trainer = Trainer()
	trainer.train()

from torch.utils.data import Dataset,DataLoader
import torch
import config as cfg

# class MyDataset(Dataset):

#     def __init__(self, dir,dir2):

#         self.dataset = []

        
#         for i, path in enumerate([dir,dir2]):
#             with open(path, "r+") as f:
#                 ws = [int(x) for x in f.readline().split()]
#                 ws_len = len(ws)
#                 start = 0
#                 while ws_len - start > cfg.pos_num:
#                     self.dataset.append((ws[start:start + cfg.pos_num ],i))
#                     start += cfg.stride
#                 else:
#                     if ws_len > cfg.pos_num :
#                         self.dataset.append([ws[ws_len - cfg.pos_num:],i])

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, index):
#         data = self.dataset[index]
#         laber =torch.tensor(data[1])
#         data = torch.tensor(data[0])
#         return data, laber

class MyDataset(Dataset):
    
    def __init__(self, dir):

        self.dataset = []

        
        
        with open(dir, "r+") as f:
            wslist =f.readlines()
            for ws in  wslist:
                data = [int(x) for x in ws.split()]
                ws_len = len(data)
                self.dataset.append((data[1:],data[0],ws_len-1))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        laber =torch.tensor(data[1])
        
        le = torch.tensor(data[2])
        data = torch.tensor(data[0]+[0 for i in range(cfg.pos_num-le)])
        p = torch.tensor([i+1 for i  in range(le)]+[0 for i in range(cfg.pos_num-le)])
        
        return data,p, laber,le-1
if __name__ == '__main__':
    myDataset = MyDataset("逆天邪神负样本token1.txt")
    dataload = DataLoader(myDataset, batch_size=10, shuffle=True)
    for i,x in  enumerate(dataload):
        # print(x)
        print(x[1])
        print(x[3])
        print(x[1][[i for i in range(x[1].size(0))],x[3]])

        break
    

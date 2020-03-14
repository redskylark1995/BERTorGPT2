import os

#根据词表将文字转化为索引
SRC_DIRS = "逆天邪神负样本.txt"#源文件地址
DST_DIR = "逆天邪神token.txt"#生成索引文件地址
VOCAB_FILE ="词表.txt"#词表位置
# if not os.path.exists(DST_DIR):
#     os.makedirs(DST_DIR)
with open(VOCAB_FILE, "r+", encoding="utf-8") as f1:
    tokens = f1.read().split()

count = 0
data = []
with open(SRC_DIRS,"r+",encoding="utf-8") as f:
    wslist = f.readlines()
    
    for ws in wslist:
        dst = ["1","1"]
        # w = f.read
        # while w:
        for w in ws:
            if w == '\n' or w == '\r' or w == '\t' or ord(w) == 12288:
                dst.append("2")
            elif w == ' ':
                dst.append("4")
            else:
                try:
                    print(w)
                    dst.append(str(tokens.index(w)))
                except:
                    # print(ord(w))
                    print("出问题了")
                    # exit()
                    dst.append("3")
        data.append(" ".join(dst)+"\r")
        print(dst)
with open(DST_DIR,"a",encoding="utf-8") as f:
    # for ws in data:

    f.writelines(data)
print("齐活了")

        
         

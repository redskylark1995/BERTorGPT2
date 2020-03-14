import os

#读取所有文字，去重后获得文字列表
txt = ""
with open("逆天邪神-c1-10.txt" ,"r",encoding='UTF-8') as R:
    while True:
        s =R.read(1024)
        if not s:
            break
        # print(s)
        txt +=s

fi = open("词表.txt","w",encoding="utf-8")
for i in ["[NULL]","[START]","[SEQ]","[UNK]","[PAD]"]:
    fi.write(i+ " ")
for i in set(list(txt)):
    if i in ["\n","\r","\t"]:
        continue
    fi.write(i+ " ")
print(len(set(list(txt))))

fi.close()

import re 
txtstr = ""
with open("逆天邪神-c1-10.txt" ,"r+",encoding='UTF-8') as R:
    while True:
        s =str(R.readline())
        if not s:
            break
        # print(s)
        txtstr +=s
        # else:break   
txtstr =re.sub("\n\r", "\n", txtstr, count=0, flags=0)
txtstr =re.sub("\r", "\n", txtstr, count=0, flags=0)
txtstr =re.sub("【.*?】", "", txtstr, count=0, flags=0)
txtstr =re.sub("投推荐票.*?报错欠更", "", txtstr, count=0, flags=0)
txtstr =re.sub("温馨提示.*?返回列表", "", txtstr, count=0, flags=0)
txtstr =re.sub("—————[\s\S]*? 第", " 第", txtstr, count=0, flags=0)
txtstr =re.sub("………[\s\S]*? 第", "第", txtstr, count=0, flags=0)
txtstr =re.sub("热门推荐.*", "", txtstr, count=0, flags=0)
for i in range(6):
    txtstr =re.sub("\n\n", "\n", txtstr, count=0, flags=0)
    txtstr =re.sub(" \n", "\n", txtstr, count=0, flags=0)


with open("逆天邪神正样本.txt","w",encoding="UTF-8") as W:
    W.write(txtstr)
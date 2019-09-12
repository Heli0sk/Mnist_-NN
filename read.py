import numpy as np
def imshow(data,mun):
    # 显示img数字
    import matplotlib.pyplot  as plt
    for i in range(mun):
        img = np.resize(data[i], (28, 28))
        plt.imshow(img)
        plt.show()
    return

def onehot(loaded_y):  # \编码
    D = {"0": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], "1": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], "2": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         "3": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], "4": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], "5": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         "6": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], "7": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], "8": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         "9": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}

    loady = []
    for i in range(len(loaded_y)):
        change = str(loaded_y[i])
        loady.append(D[(change)])
    return loady

def readdata():
    '''
    读取mnist图像数据 类似bmp文件
    :return: 数据/流
    '''
    # 加载二进制数据
    with open('./data/train-images-idx3-ubyte','rb') as x :
        with open('./data/train-labels-idx1-ubyte', 'rb') as  y :
            # 解析
            x.read(16)#所在的位置开始读16个字节
            y.read(8)
            loaded_x = np.fromfile(x, dtype=np.uint8,count=-1)/255
            loaded_x = np.reshape(loaded_x,[-1,28,28,1])
            loaded_y = np.fromfile(y, dtype=np.uint8)
    #张量流

    print('load complete!')
    loady = onehot(loaded_y)
    return loaded_x[:55000] ,loady[:55000],loaded_x[55000:60000],loady[55000:60000]

if __name__ == '__main__':

    a,b ,c,d =  readdata()
    imshow(a,10)
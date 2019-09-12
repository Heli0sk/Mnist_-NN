import numpy as np

def imshow(img,lable,mun):
    """
    # 显示img数字
    :param data: image
    :param mun: show num
    :return: 0
    """
    import matplotlib.pyplot  as plt
    for i in range(mun):
        img1 = np.resize(img[i], (28, 28))
        plt.imshow(img1)
        print(lable[i])
        plt.show()
    return

def onehot(loaded_y):  # \编码  class
    """
    :param loaded_y:
    :return: coded
    """
    D = {"0": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], "1": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], "2": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         "3": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], "4": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], "5": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         "6": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], "7": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], "8": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         "9": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}

    loady = list()
    for i in range(len(loaded_y)):
        change = str(loaded_y[i])
        loady.append(D[(change)])
    return loady

def readdata():
    """
      读取mnist图像数据 类似bmp文件
    :return: 数据/流
    """
    # 加载二进制数据
    def load_x(dir):
        x = open(dir,"rb")
        x.read(16)  # 所在的位置开始读16个字节
        loaded_x = np.fromfile(x, dtype=np.uint8, count=-1) / 255
        loaded_x = np.reshape(loaded_x, [-1, 28, 28, 1])
        x.close()
        return loaded_x

    def load_y(dir):
        with open(dir, 'rb') as y:# 解析
            y.read(8)#所在的位置8个 offset 开始字节
            y = np.fromfile(y, dtype=np.uint8)
        return y

    loaded_x = load_x("./data/train-images-idx3-ubyte")
    loaded_xte = load_x("./data/t10k-images-idx3-ubyte")
    loaded_y = load_y('./data/train-labels-idx1-ubyte')
    loaded_yte = load_y('./data/t10k-labels-idx1-ubyte')

    loaded_y = onehot(loaded_y)
    loaded_yte = onehot(loaded_yte)

    print('load complete!')
    return loaded_x ,loaded_y,loaded_xte,loaded_yte

if __name__ == '__main__':

    a,b ,c,d =  readdata()
    print(len(a))
    print(len(b))
    print(len(c))
    print(len(d))
    imshow(a,b,2)
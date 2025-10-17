import json
import matplotlib.pyplot as plt
import numpy as np

def drawFromJson(path):
    with open(path, 'r') as file:
        loss_list = json.load(file)
    loss_array = np.array(loss_list)
    x_epoch = np.array(loss_array[:, 0])
    y_trainloss = np.array(loss_array[:, 1])
    y_valloss = np.array(loss_array[:, 2])
    y_valiou = np.array(loss_array[:, 3])
    plt.plot(x_epoch, y_trainloss)
    plt.plot(x_epoch, y_valloss)
    plt.plot(x_epoch, y_valiou)
    plt.xlabel('epoch')
    plt.ylabel('loss/iou')
    plt.grid()
    plt.legend(['train_loss', 'val_loss', 'val_iou'])
    plt.show()

if __name__ == '__main__' :
    drawFromJson('models/lr3e-4_noF/loss.json')
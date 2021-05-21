import glob

import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models


def predict(input, model, device):
    model.to(device)
    with torch.no_grad():
        input = input.to(device)
        out = model(input)
        _, pre = torch.max(out.data, 1)
        return pre.item()


if __name__ == "__main__":
    """ 
    这个是将所有的label填写到一个列表里面，然后将这个label的列表转化为数字
    最后在csv的文件中会再次将数字对应回去
    ['plastic', 'paper', 'cardboard', 'trash', 'glass', 'metal']
    """
    model = models.resnet101(pretrained=False)
    fc_inputs = model.fc.in_features
    model.fc = nn.Linear(fc_inputs, 6)
    model = model.cuda()
    # 加载训练好的模型
    checkpoint = torch.load('model_best_checkpoint_resnet101.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    test_images_dir = "./test/*.jpg"
    images = glob.glob(test_images_dir)
    images_sorted = sorted(images, key=lambda x: (x.split('.')[1][15:]))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
    #
    status_dict = ['plastic', 'paper', 'cardboard', 'trash', 'glass', 'metal']
    filename_list = []
    label_pred_list = []
    for i in images_sorted:
        img = Image.open(i)
        img = test_tf(img).unsqueeze(0)
        ans = predict(img, model, device)
        print("{}的预测结果是{}".format(i, status_dict[ans]))
        filename = i.split('.')[1][15:]
        filename_list.append(filename)
        label_pred_list.append(ans)
    data = list(zip(filename_list, label_pred_list))
    df = pd.DataFrame(data=data, columns=['filename', 'label_nums'])
    df['label'] = df['label_nums'].apply(lambda x: status_dict[x])
    df.drop('label_nums',axis=1,inplace=True)
    df.to_csv('./pred1_resnet101_0829.csv', index=False, header=False)

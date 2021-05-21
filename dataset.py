import torch
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True  # 跳过损坏图


class Garbage_lader(Dataset):
    def __init__(self, csv_path, train_flag=True):
        self.imgs_info = self.get_images(csv_path)
        self.train_flag = train_flag

        self.train_tf = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ]
        )
        self.val_tf = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
            ]
        )

    def get_images(self, cv_path):
        train_df = pd.read_csv(cv_path)
        files = train_df.filename
        labels_kinds = train_df.label.unique().tolist()  # unique获取唯一值
        images_path = ["D:/pytorchtest/garbage/train/" + i for i in files]
        labels = train_df.label.apply(lambda x: labels_kinds.index(x))
        labels = labels.tolist()
        images_info = dict(zip(images_path, labels))  # 路径和label合成一个元祖
        images_info = list(images_info.items())
        return images_info

    def padding_black(self, img):
        w, h = img.size
        scale = 224 / max(w, h)
        img_fg = img.resize([int(x) for x in [w * scale, h * scale]])

        size_fg = img_fg.size
        size_bg = 224

        img_bg = Image.new("RGB", (size_bg, size_bg))

        img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                              (size_bg - size_fg[1]) // 2))

        img = img_bg
        return img

    def __getitem__(self, index):
        img_path, label = self.imgs_info[index]
        img = Image.open(img_path)
        img = img.convert("RGB")
        img = self.padding_black(img)
        if self.train_flag:
            img = self.train_tf(img)

        else:
            img = self.val_tf(img)
        label = int(label)

        return img, label

    def __len__(self):
        return len(self.imgs_info)


if __name__ == "__main__":
    train_dataset = Garbage_lader('D:/pytorchtest/garbage/train.csv', True)
    print("数据个数：", len(train_dataset))
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=True
    )

    for img, label in train_loader:
        print(img.shape)
        print(label)

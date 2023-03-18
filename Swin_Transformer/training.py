import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from timm.scheduler.cosine_lr import CosineLRScheduler
import time

from Swin_Transformers.main import Swin_Transformer

class Swin_Transformer_classification:
    def __init__(self, model, train_dir, valid_dir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.iteration = 300
        self.model = model.to(self.device)

        self.train_dataset = ImageFolder(train_dir, self.augmentation()[0])
        self.valid_dataset = ImageFolder(valid_dir, self.augmentation()[1])

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=256, shuffle=True)
        self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=128, shuffle=False)

        num_steps = self.iteration * len(self.train_dataloader)
        warm_steps = 20 * len(self.train_dataloader)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.05)

        self.lr_scheduler = CosineLRScheduler(
            optimizer=self.optimizer,
            t_initial=num_steps - warm_steps,
            # T_mult=1.,
            lr_min=5e-6,
            warmup_lr_init=5e-7,
            warmup_t=warm_steps,
            t_in_epochs=False,
            warmup_prefix=True
        )

    def augmentation(self):
        train_trans = transforms.Compose([
            transforms.Resize([512, 512]),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        valid_trans = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return train_trans, valid_trans

    def model_training(self, num_update):
        self.model.train()
        training_loss = 0
        training_corrects = 0
        for inputs, labels in self.train_dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step_update(num_updates=num_update)

            training_loss += loss.item() * inputs.size(0)
            training_corrects += torch.sum(preds == labels.data)

        return training_loss, training_corrects

    def model_validating(self):
        validating_loss = 0
        validating_corrects = 0

        self.model.eval()
        with torch.no_grad():
            for inputs, labels in self.valid_dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                validating_loss += loss.item() * inputs.size(0)
                validating_corrects += torch.sum(preds == labels.data)

            return validating_loss, validating_corrects

    def training(self):
        loss = list() # loss값 확인용
        acc = list() # accuracy값 확인 용
        start = time.time()

        for epoch in range(self.iteration):
            num_steps_per_epoch = len(self.train_dataloader)
            num_updates = epoch * num_steps_per_epoch

            training_loss, training_corrects = self.model_training(num_updates)

            train_loss = training_loss / len(self.train_dataset)
            train_acc = training_corrects / len(self.train_dataset) * 100

            print('#{} Train Loss: {:.4f} Train Acc: {:.4f}% Time: {:.4f}s'.format(epoch, train_loss, train_acc,time.time() - start))

            valid_loss, valid_corrects = self.model_validating()
            valid_loss = valid_loss / len(self.val_dataset)
            valid_acc = valid_corrects / len(self.val_dataset) * 100

            print('#{} Valid Loss: {:.4f} Valid Acc: {:.4f}% Time: {:.4f}s'.format(epoch, valid_loss, valid_acc, time.time() - start))

            loss.append(valid_loss)
            acc.append(valid_acc)

        return loss, acc, model

# 학습할거 정하고, 모델 타입, 패치 사이즈
model = Swin_Transformer("classification", "t", 4)

s = Swin_Transformer_classification(model, "../dataset/imagenet-mini/train","../dataset/imagenet-mini/val")
loss, acc, model = s.training()
torch.save({"model":model.load_state_dict(),
            "loss":loss,
            "accuracy":acc
            }, "./model/swin_classfication.tar")


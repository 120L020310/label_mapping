# coding:utf-8
import numpy as np
import torch
import random
import logging
import numpy as np
import argparse
import os
import sys
from datetime import datetime
import math
import time

import torchvision
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import models
from torchvision import transforms
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import utils4

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

utils4.set_logger('train.log')

# Create the input data pipeline
parser = argparse.ArgumentParser()

parser.add_argument("--total_epoch", default=35, type=int, help="Total number of training epochs to perform.")
parser.add_argument('--train_dir', default='D:\\label_mapping\\src_data\\datas\\train_pre', help="Directory containing the dataset")
parser.add_argument('--test_dir', default='D:\\label_mapping\\src_data\\datas\\test_pre',help="Directory containing the dataset")
parser.add_argument('--model_dir', default='D:\\label_mapping\\model_dir',help="Directory containing params.json")
parser.add_argument('--restore_file', default="best_67_4", help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--charset_size', default=67, help="Choose the first `charset_size` character to conduct our "
                                                        "experiment.")
parser.add_argument('--image_size', default=112, help="Needs to provide same value as in training.")
parser.add_argument('--max_steps', default=3000000000, help='the max training steps ')
parser.add_argument('--eval_steps', default=34573, help="the step num to eval")
parser.add_argument('--save_steps', default=34573, help="the steps to save")
parser.add_argument('--restore', default=True, help='whether to restore from checkpoint')
parser.add_argument('--epoch', default=30, help='Number of epoches')
parser.add_argument('--decay_steps', default=34573, help='')
parser.add_argument('--batch_size', default=256, help='')
parser.add_argument('--mode', choices=["train", "validation", "inference"], default='train', help='Running mode')

args = parser.parse_args()
args.device = device

logging.info("-----------------------------main.py start--------------------------")


def load_data(args):
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),  # 调整图像大小
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化图像张量
    ])
    logging.info("开始加载数据...")
    train_dataset = torchvision.datasets.ImageFolder(root=args.train_dir, transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(root=args.test_dir, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    return train_loader,test_loader
acc = []
result = []
step = 0

 ########################  后续需要在循环里scheduler.step()######################

def eval(args, model, test_loader):
    eval_loss = 0.0
    total_acc = 0.0
    correct_top5 = 0.0
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()
    num = 0
    loss_list=[]
    labels_list=[]
    preds_list=[]
    total = 0
    for images, l in tqdm(test_loader):
        # num1=num1+1
        images = images.to(device)
        labels = map(l, 4)
        # end=time.time()
        # t=end-start
        labels = torch.round(labels)
        labels = labels.long()
        labels = labels.to(device)
        total += labels.size(0)
        with torch.no_grad():
            logits = model(images) # model返回的是（bs,num_classes）和weight
            batch_loss = loss_function(logits, labels)
            # 记录误差
            eval_loss += batch_loss.item()
            # 记录准确率
            _, preds = logits.max(1)
            _1, pred_top5 = logits.topk(5, 1, largest=True, sorted=True)
            num_correct = (preds == labels).sum().item()
            # print("pred:",preds)
            # print("predtop5:",pred_top5)
            total_acc += num_correct
            acc_top5 = torch.eq(pred_top5, labels.view(-1, 1)).sum().item()  # 计算匹配的数量
            correct_top5 = correct_top5 + acc_top5
        preds = preds.cpu().numpy()
        loss = batch_loss.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        loss_list.append(loss)
        labels_list.extend(labels)
        preds_list.extend(preds)

    log_test = {}
    # 计算分类评估指标
    acc_top5 = 100 * correct_top5 / total
    log_test['test_loss'] = np.mean(loss_list)
    log_test['test_accuracy'] = accuracy_score(labels_list, preds_list)
    log_test['test_accuracy_top5'] = acc_top5
    log_test['test_precision'] = precision_score(labels_list, preds_list, average='macro')
    log_test['test_recall'] = recall_score(labels_list, preds_list, average='macro')
    log_test['test_f1-score'] = f1_score(labels_list, preds_list, average='macro')
    wandb.log(log_test)
    loss = eval_loss / len(test_loader)
    acc = total_acc / (len(test_loader) * args.batch_size)
    return loss, acc,acc_top5

def map(label,i):
    l=label % 67
    c=label/67
    f=torch.floor(c)
    label=(l+f*i)%67
    return label

def train(args, model,train_loader,test_loader,optimizer,criterion,scheduler,restore_file=None):
    iters=0
    if restore_file is not None:
        restore_path = os.path.join(
            args.model_dir, restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils4.load_checkpoint(restore_path, model, optimizer)
        eval_loss0, eval_acc0, eval_acc_top50 = eval(args, model, test_loader)
        best_val_acc = eval_acc0
    else:
        best_val_acc = 0.0
    print("best_val_acc:", best_val_acc)
    global step
    #torch.cuda.set_per_process_memory_fraction(0.8)  # 设置每个进程的GPU显存分配比例为0.8
    # 创建PyTorch会话
    torch.cuda.empty_cache()
    logging.info("开始训练.........................")
    # 设置测试损失list,和测试acc 列表
    val_loss_list = []
    val_acc_list = []
    # 设置训练损失list
    train_loss_list = []
    length = len(train_loader)
    for i in range(args.total_epoch):
        model.train()
        train_loss = 0
        #num=0
        # times=0.0
        # train_times=0.0
        for images, l in tqdm(train_loader):
            #num=num+1
            #start=time.time()
            labels = map(l, 4)
            #end=time.time()
            #t=end-start
            labels = torch.round(labels)
            labels = labels.long()
            images = images.to(device)
            # labels = labels % 67
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_loss += loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # 获取当前batch的标签类别和预测类别
            _, preds = torch.max(outputs, 1)  # 获得当前batch 所有图像的预测类别preds =preds. cpu( ).numpy()
            preds=preds.cpu().numpy()

            loss = loss.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            log_train = {}
            log_train['epoch'] = args.total_epoch
            log_train['train_loss'] = loss
            log_train['train_accuracy'] = accuracy_score(labels, preds)
            wandb.log(log_train)
            # over=time.time()
            # train_time=over-end
            # print("第"+str(i)+"个epoch:"+str(num)+"/"+str(length)+", loss="+str(train_loss))
            #if num==15:break
            #     print(times)
            #     print(train_times)
            # times+=t
            # train_times+=train_time
        # 每训练一个epoch,记录一次训练损失
        train_loss = train_loss / len(train_loader)
        train_loss_list.append(train_loss)
        print("train Epoch:{},loss:{}".format(i, train_loss))

        # 每训练一个epoch,用当前训练的模型对验证集进行测试
        eval_loss, eval_acc,eval_acc_top5 = eval(args, model, test_loader)
        # 将每一个测试集验证的结果加入列表
        val_loss_list.append(eval_loss)
        val_acc_list.append(eval_acc)

        print("val Epoch:{},eval_loss:{},eval_acc:{},top5_acc:{}".format(i, eval_loss, eval_acc,eval_acc_top5))
        scheduler.step()

        #  判断是否需要保存模型
        is_best = eval_acc >= best_val_acc
        if(is_best): best_val_acc=eval_acc
        print("is_best=",is_best,"best_val_acc=",best_val_acc)
        # 每个epcoh保存一次模型参数
        print("start save model......")
        # Save weights
        utils4.save_checkpoint({'epoch': i + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=args.model_dir)
        print("finish save model......")

wandb.init(project='resnet_67_4', name=time.strftime('%m%d%H%M%S'))
net=models.resnet50(pretrained=True)
net.fc = nn.Linear(net.fc.in_features, 67)
# model1.add_module("fc1",nn.Linear(model1.fc.out_features, 3755))

#
learning_rate = 0.001
decay_steps = args.decay_steps
decay_rate = 0.97
# 创建优化器
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
# 创建学习率衰减器
scheduler = ExponentialLR(optimizer, gamma=decay_rate)
criterion = nn.CrossEntropyLoss()
train_loader,test_loader=load_data(args)
net = net.to(device)

train(args,net,train_loader,test_loader,optimizer,criterion,scheduler,args.restore_file)
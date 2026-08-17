from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from training.data import CornDataset
from pct.model import Pct
import numpy as np
from torch.utils.data import DataLoader
from pct.util import cal_loss, IOStream
import sklearn.metrics as metrics
import time
import shutil  # 确保导入shutil


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')

    try:
        shutil.copy('main.py', 'checkpoints/' + args.exp_name + '/main.py.backup')
        shutil.copy('model.py', 'checkpoints/' + args.exp_name + '/model.py.backup')
        shutil.copy('util.py', 'checkpoints/' + args.exp_name + '/util.py.backup')
        shutil.copy('data.py', 'checkpoints/' + args.exp_name + '/data.py.backup')
    except Exception as e:
        print(f"Warning: Failed to backup files. {e}")


def train(args, io):
    if args.dataset == 'modelnet40':
        NUM_CLASSES = 40
    elif args.dataset == 'corn':
        NUM_CLASSES = 2
    else:
        NUM_CLASSES = 2

    train_loader = DataLoader(CornDataset(partition='train', num_points=args.num_points, data_dir=args.data_dir), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(CornDataset(partition='test', num_points=args.num_points, data_dir=args.data_dir), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    model = Pct(args, output_channels=NUM_CLASSES).to(device)
    print(str(model))
    model = nn.DataParallel(model)

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=5e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    criterion = cal_loss
    best_test_acc = 0
    start_epoch = 0

    # --- 新增/修改部分：初始化 scheduler ---
    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)

    # --- 修改部分：加载模型状态 ---
    if args.model_path != '':
        print(f'Loading checkpoint from {args.model_path}')
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)

        # 加载模型权重
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # 兼容只保存了模型权重的情况
            model.load_state_dict(checkpoint)
            print("Warning: Checkpoint does not contain 'state_dict' key. Assuming whole checkpoint is state_dict.")

        # 加载优化器状态
        if 'optimizer' in checkpoint and not args.eval:
            opt.load_state_dict(checkpoint['optimizer'])
            print("Loaded optimizer state.")
        else:
            print("Warning: Checkpoint does not contain 'optimizer' state.")

        # 加载起始轮次
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1  # 从下一轮开始
            print(f"Resuming training from epoch {start_epoch}")
        else:
            print("Warning: Checkpoint does not contain 'epoch' info. Starting from epoch 0.")

        # --- 新增部分：加载 scheduler 状态 ---
        if 'scheduler' in checkpoint and not args.eval:
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("Loaded scheduler state.")
        else:
            print("Warning: Checkpoint does not contain 'scheduler' state. Scheduler will start fresh.")

        # 加载最佳准确率
        if 'best_acc' in checkpoint:
            best_test_acc = checkpoint['best_acc']
            print(f"Best test accuracy loaded: {best_test_acc:.6f}")

    # 修改：从 start_epoch 开始训练
    for epoch in range(start_epoch, args.epochs):
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        idx = 0
        total_time = 0.0
        for data, label in (train_loader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()

            start_time = time.time()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            end_time = time.time()
            total_time += (end_time - start_time)

            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
            idx += 1

        scheduler.step()
        print('train total time is', total_time)
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss * 1.0 / count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        total_time = 0.0
        with torch.no_grad():  # 测试时关闭梯度计算
            for data, label in test_loader:
                data, label = data.to(device), label.to(device).squeeze()
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                start_time = time.time()
                logits = model(data)
                end_time = time.time()
                total_time += (end_time - start_time)
                loss = criterion(logits, label)
                preds = logits.max(dim=1)[1]
                count += batch_size
                test_loss += loss.item() * batch_size
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())

        print('test total time is', total_time)
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss * 1.0 / count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)

        # --- 修改部分：保存完整的训练状态 ---
        # 保存最新的模型，用于断点续训
        latest_checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': opt.state_dict(),
            'scheduler': scheduler.state_dict(),  # 新增
            'best_acc': best_test_acc,
            'args': args
        }
        torch.save(latest_checkpoint, 'checkpoints/%s/models/latest_model-new.t7' % args.exp_name)

        # 如果是最佳模型，再保存一份
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            best_checkpoint = latest_checkpoint.copy()
            torch.save(best_checkpoint, 'checkpoints/%s/models/model-new.t7' % args.exp_name)
            print(f"Best model updated and saved at epoch {epoch}.")


def test(args, io):
    # 根据数据集设置类别数
    if args.dataset == 'modelnet40':
        NUM_CLASSES = 40
    elif args.dataset == 'corn':
        NUM_CLASSES = 2
    else:
        NUM_CLASSES = 2

    test_loader = DataLoader(CornDataset(partition='test', num_points=args.num_points, data_dir=args.data_dir),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    model = Pct(args, output_channels=NUM_CLASSES).to(device)
    model = nn.DataParallel(model)

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.eval()
    test_true = []
    test_pred = []

    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            logits = model(data)
            preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Final Test :: test acc: %.6f, test avg acc: %.6f' % (test_acc, avg_per_class_acc)
    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='corn_pct_finetune', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='corn', metavar='N',
                        choices=['modelnet40', 'corn'],
                        help='Name of the dataset (default: corn)')
    parser.add_argument('--batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action='store_true', default=False,
                        help='Use SGD (default: False, which means use Adam)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=4096,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='dropout rate')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory containing ply_data_train.h5 and ply_data_test.h5; defaults to CORN_DATA_DIR or data/corn')
    args = parser.parse_args()

    # 确保日志目录存在
    log_dir = os.path.join('checkpoints', args.exp_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 初始化日志
    io = IOStream(os.path.join(log_dir, 'corn-run.log'))
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        _init_()
        train(args, io)
    else:
        if args.model_path == '':
            print("Error: --model_path must be specified for evaluation.")
        else:
            test(args, io)
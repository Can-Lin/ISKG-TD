import pandas as pd
import time
from math import log2
from ISKG_model import ISKGModel
from utils import fix_seed_torch, draw_loss_pic
import argparse
from Logger import Logger
from mydataset import MyDataset, ISKGTDDataset
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
import sys

# 固定随机数种子
fix_seed_torch(seed=2021)
# 设置训练的超参数
parser = argparse.ArgumentParser()
parser.add_argument('--gcn_layers', type=int, default=2, help='the number of gcn layers')
parser.add_argument('--n_epochs', type=int, default=50, help='the number of epochs')
parser.add_argument('--embedSize', type=int, default=64, help='dimension of user and entity embeddings')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--ratio', type=float, default=0.8, help='size of training dataset')
parser.add_argument('--pretrained', type=bool, default=False, help='use pretrained model')
args = parser.parse_args()
# 设备是否支持cuda
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
args.device = device
print(args.device)
# 读取用户特征、天气特征、评分
user_feature = pd.read_csv('./yelp_data/yelp_user.csv')
knowledge_feature = pd.read_csv('./yelp_data/yelp_business_category.csv')
social_feature = pd.read_csv('./yelp_data/yelp_user_user.csv')
item_feature = pd.read_csv('./yelp_data/yelp_business.csv')
rating = pd.read_csv('./yelp_data/yelp_user_business.csv')
testrating = pd.read_csv('./yelp_data/yelp_test_data.csv')
# 构建数据集
dataset = ISKGTDDataset(rating)
trainLen = int(args.ratio * len(dataset))
train, test = random_split(dataset, [trainLen, len(dataset) - trainLen])
train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(test, batch_size=len(test))
# 记录训练的超参数
start_time = '{}'.format(time.strftime("%m-%d-%H-%M", time.localtime()))
logger = Logger('./log/log-{}.txt'.format(start_time))
logger.info(' '.join('%s: %s' % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
# 定义模型
model = ISKGModel(args, user_feature, item_feature, social_feature, knowledge_feature, rating)
model.to(device)
# 定义优化器
optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.001)
# 定义损失函数
loss_function = MSELoss()
train_result = []
test_result = []
# 最好的epoch
best_loss = sys.float_info.max
# 'sys.float_info.max'是Python语言中的一个浮点数常量，表示机器可表示的最大浮点数。

if not args.pretrained:
    # 训练
    for i in range(args.n_epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            prediction = model(batch[0].to(device), batch[1].to(device))
            train_loss = torch.sqrt(loss_function(batch[2].float().to(device), prediction))
            # 反向传播，计算梯度
            train_loss.backward()
            # 使用优化器更新模型参数
            optimizer.step()
        train_result.append(train_loss.item())
        model.eval()
        for data in test_loader:
            prediction = model(data[0].to(device), data[1].to(device))
            test_loss = torch.sqrt(loss_function(data[2].float().to(device), prediction))
            test_loss = test_loss.item()
            if best_loss > test_loss:
                best_loss = test_loss
                torch.save(model.state_dict(), './ISKG_model/bestModeParms-{}.pth'.format(start_time))
            test_result.append(test_loss)
            logger.info("Epoch{:d}:trainLoss{:.4f},testLoss{:.4f}".format(i, train_loss, test_loss))
    model.load_state_dict(torch.load('./ISKG_model/bestModeParms-{}.pth'.format(start_time)))
else:
    model.load_state_dict(torch.load("./ISKG_model/bestModeParms-05-16-22-28.pth"))

# 画图
draw_loss_pic(train_result, test_result)
# 加载最佳模型
model.eval()
total = 0
predicte_score_true = 0
totalarrays = []

user_tensor = torch.tensor(testrating['user_id'].values).to(device)
business_tensor = torch.tensor(testrating['business_id'].values).to(device)
y_true = model(user_tensor, business_tensor)
y_scores = torch.tensor(testrating['stars'].values).to(device)
y_true = y_true.long()
# 将预测得分映射到五分制评分，例如，使用0.5作为阈值来决定是否推荐
y_pred = torch.round(y_scores.float()).long()  # 映射到整数，并四舍五入
# 计算Recall
# 注意：这里我们假设推荐列表长度为k，你可以根据实际情况修改k的值
k = 10
recall_value = torch.sum(y_true & y_pred).float() / torch.sum(y_true)
# 计算NDCG
def ndcg_at_k(y_true, y_scores, k=10):
    # 计算DCG
    dcg = sum(1 / log2(i + 2) for i, score in enumerate(y_scores) if y_true[i] == 1)
    # 计算IDCG
    idcg = sum(1 / log2(i + 2) for i, score in enumerate(y_scores) if i < min(k, len(y_true)))
    # 计算NDCG
    return dcg / idcg if idcg > 0 else 0
# 计算NDCG
ndcg_value = ndcg_at_k(y_true, y_scores)
# 输出结果
print(f"Recall: {recall_value}")
print(f"NDCG: {ndcg_value}")

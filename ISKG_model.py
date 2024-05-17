import torch.nn as nn
from utils import *
from torch.nn import Module
import scipy.sparse as sp

class GCN_Layer(Module):
    def __init__(self, inF, outF):
        super(GCN_Layer, self).__init__()
        self.W1 = torch.nn.Linear(in_features=inF, out_features=outF)
        self.W2 = torch.nn.Linear(in_features=inF, out_features=outF)

    def forward(self, graph, selfLoop, features):
        part1 = self.W1(torch.sparse.mm(graph + selfLoop, features))
        part2 = self.W2(torch.mul(torch.sparse.mm(graph, features), features))
        return nn.LeakyReLU()(part1 + part2)

class ISKGModel(Module):
    def __init__(self, args, user_feature, item_feature, social_feature, knowledge_feature, rating):
        super(ISKGModel, self).__init__()
        self.args = args
        self.device = args.device
        self.user_feature = user_feature
        self.item_feature = item_feature
        self.social_feature = social_feature
        self.knowledge_feature = knowledge_feature
        self.rating = rating
        self.num_user = len(rating['user_id'].unique())
        self.num_item = len(rating['business_id'].unique())
        # user embedding
        self.user_id_embedding = nn.Embedding(user_feature['user_id'].max() + 1, 32)
        self.user_useful_embedding = nn.Embedding(user_feature['useful'].max() + 1, 4)
        self.user_funny_embedding = nn.Embedding(user_feature['funny'].max() + 1, 4)
        self.user_cool_embedding = nn.Embedding(user_feature['cool'].max() + 1, 4)
        self.user_fans_embedding = nn.Embedding(user_feature['fans'].max() + 1, 4)
        self.user2_id_embedding = nn.Embedding(social_feature['user2_id'].max() + 1, 4)
        # Knowledge embedding
        self.knowledge_embedding = nn.Embedding(knowledge_feature['business_id'].max() + 1, 8)
        # item embedding
        self.item_id_embedding = nn.Embedding(item_feature['business_id'].max() + 1, 32)
        self.item_review_count_embedding = nn.Embedding(item_feature['review_count'].max() + 1, 12)
        # 自循环
        self.selfLoop = self.getSelfLoop(self.num_user + self.num_item)
        # 堆叠GCN层
        self.GCN_Layers = torch.nn.ModuleList()
        for _ in range(self.args.gcn_layers):
            self.GCN_Layers.append(GCN_Layer(self.args.embedSize, self.args.embedSize))
        self.graph = self.buildGraph()
        self.transForm = nn.Linear(in_features=self.args.embedSize * (self.args.gcn_layers + 1),
                                   out_features=self.args.embedSize)


    def float_embedding(self, float_values):
        # 规范化到0到1
        normalized_values = (float_values - float_values.min()) / (float_values.max() - float_values.min())
        # 量化到0到1000
        quantized_indices = np.floor(normalized_values * 1000).astype(int)
        # 创建嵌入层
        embedding = nn.Embedding(num_embeddings=1001, embedding_dim=12)  # 1001因为索引到1000
        # 将索引转换为tensor
        indices_tensor = torch.tensor(quantized_indices, dtype=torch.long)
        # 获取嵌入向量
        embedded_vectors = embedding(indices_tensor).to(self.device)
        return embedded_vectors

    # 自环，是否考虑自身节点信息
    def getSelfLoop(self, num):
        i = torch.LongTensor(
            [[k for k in range(0, num)], [j for j in range(0, num)]])
        val = torch.FloatTensor([1] * num)
        return torch.sparse_coo_tensor(i, val).to(self.device)

    def buildGraph(self):
        # 构建链接矩阵
        rating = self.rating.values
        graph = sp.coo_matrix(
            (rating[:, 2], (rating[:, 0], rating[:, 1])), shape=(self.num_user, self.num_item), dtype=int).tocsr()
        graph = sp.bmat([[sp.csr_matrix((graph.shape[0], graph.shape[0])), graph],
                         [graph.T, sp.csr_matrix((graph.shape[1], graph.shape[1]))]])

        # 拉普拉斯变化
        row_sum_sqrt = sp.diags(1 / (np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
        col_sum_sqrt = sp.diags(1 / (np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
        # @ 在Python中表示矩阵乘法
        graph = row_sum_sqrt @ graph @ col_sum_sqrt
        graph = graph.tocoo()
        # 使用torch的稀疏张量表示
        values = graph.data
        indices = np.vstack((graph.row, graph.col))
        graph = torch.sparse_coo_tensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))
        return graph.to(self.device)    # name, review_count, yelping_since

    def getFeature(self):
        # 根据用户特征获取对应的embedding
        user_id = self.user_id_embedding(torch.tensor(self.user_feature['user_id']).to(self.device))
        useful = self.user_useful_embedding(torch.tensor(self.user_feature['useful']).to(self.device))
        funny = self.user_funny_embedding(torch.tensor(self.user_feature['funny']).to(self.device))
        cool = self.user_cool_embedding(torch.tensor(self.user_feature['cool']).to(self.device))
        fans = self.user_fans_embedding(torch.tensor(self.user_feature['fans']).to(self.device))
        average_stars = self.float_embedding(self.user_feature['average_stars'])
        user1_social_features = []
        for user1_id in self.social_feature['user1_id'].unique():
            user2_ids = self.social_feature[self.social_feature['user1_id'] == user1_id]['user2_id'].values
            user2_embeddings = self.user2_id_embedding(torch.tensor(user2_ids).to(self.device))
            user1_social_feature = user2_embeddings.mean(dim=0)  # 计算平均嵌入
            user1_social_features.append(user1_social_feature)
        user1_social_features = torch.stack(user1_social_features)
        user_emb = torch.cat((user_id, useful, funny, cool, fans, average_stars, user1_social_features), dim=1)
        item_knowledge_features = []
        for item_id in self.knowledge_feature['business_id'].unique():
            # 获取与当前item_id相关的所有entity_id
            entity_ids = self.knowledge_feature[self.knowledge_feature['business_id'] == item_id]['category_id'].values
            # 为这些entity_id生成嵌入
            entity_embeddings = self.knowledge_embedding(torch.tensor(entity_ids).to(self.device))
            # 计算这些entity_id嵌入的平均值
            item_knowledge_feature = entity_embeddings.mean(dim=0)
            item_knowledge_features.append(item_knowledge_feature)

        item_knowledge_features = torch.stack(item_knowledge_features)
        # 现在，您可以将item_knowledge_features与item_id的其他特征进行拼接
        item_id = self.item_id_embedding(torch.tensor(self.item_feature['business_id']).to(self.device))
        item_stars = self.float_embedding(self.item_feature['stars'])
        item_review_count = self.item_review_count_embedding(torch.tensor(self.item_feature['review_count']).to(self.device))
        # 拼接item_id的其他特征与知识特征
        item_emb = torch.cat((item_id, item_stars, item_review_count, item_knowledge_features), dim=1)
        # 拼接到一起
        concat_emb = torch.cat([user_emb, item_emb], dim=0)
        return concat_emb.to(self.device)

    def forward(self, users, items):
        #features,social1,social2 = self.getFeature()
        features = self.getFeature()
        # clone() 返回一个和源张量同shape、dtype和device的张量，与源张量不共享数据内存，但提供梯度的回溯。
        final_emb = features.clone()
        for GCN_Layer in self.GCN_Layers:
            features = GCN_Layer(self.graph, self.selfLoop, features)
            final_emb = torch.cat((final_emb, features.clone()), dim=1)
        user_emb, item_emb = torch.split(final_emb, [self.num_user, self.num_item])
        user_emb = user_emb[users]
        item_emb = item_emb[items]
        user_emb = self.transForm(user_emb)
        item_emb = self.transForm(item_emb)
        prediction = torch.mul(user_emb, item_emb).sum(1)
        return prediction

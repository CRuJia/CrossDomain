import torch.nn as nn
from abc import abstractmethod
import numpy as np


class MetaTemplate(nn.Module):
    def __init__(self, model_func, n_way, n_support, n_query, use_cuda=False):
        super(MetaTemplate, self).__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query #TODO (change depends on input)
        self.use_cuda = use_cuda
        self.feature = model_func
        self.feat_dim = self.feature.final_feat_dim

    @abstractmethod
    def set_forward(self, x, is_feature):
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        pass

    def forward(self, x):
        out = self.feature.forward(x)
        return out

    def parse_feature(self, x, is_feature):
        if self.use_cuda:
            x = x.cuda()
        if is_feature:
            z_all = x
        else:
            x = x.contiguous().view(self.n_way * (self.n_support + self.n_query), *x.size()[2:]) #(n_way,n_support+n_query,c,w,h) -> (n_way*(n_support+n_query),c,w,h)
            z_all = self.feature.forward(x)
            z_all = z_all.view(self.n_way, self.n_support+self.n_query, -1)

        # print(z_all.shape)
        z_support = z_all[:, :self.n_support]
        z_query = z_all[:, self.n_support:]
        # print("z_support", z_support.shape)
        # print("z_query", z_query.shape)


        return z_support,z_query

    def correct(self,x):
        scores, loss = self.set_forward_loss(x)
        y_query = np.repeat(range(self.n_way), self.n_query)

        topk_scores, topk_labels = scores.data.topk(1,1,True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:,0] == y_query)
        return float(top1_correct), len(y_query), loss.item()*len(y_query)



    def test_loop(self, test_loader, recoder=None):
        loss = 0.
        count = 0
        acc_all = []
        iter_num = len(test_loader)
        for i,(x,_) in enumerate(test_loader):
            correct_this, count_this, loss_this = self.correct(x)
            acc_all.append(correct_this/ count_this*100)
            loss += loss_this
            count += count_this

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print("--- %d Loss %.6f ---"%(iter_num, loss/count))
        print("--- %d Test Acc = %4.2f%% +- %4.2f%% ---"%(iter_num,acc_mean, 1.96*acc_std/np.sqrt(iter_num)))

        return acc_mean







import torch
import torch.nn as nn
import numpy as np
from models.meta_template import MetaTemplate
from models import backbone
from torchvision import models

class ProtoNet(MetaTemplate):
    def __init__(self,model_func, n_way, n_support, n_query):
        super(ProtoNet, self).__init__(model_func, n_way, n_support, n_query, use_cuda=False)
        self.loss_fn = nn.CrossEntropyLoss()
        self.method = 'ProtoNet'

    def reset_modules(self):
        return

    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)
        z_support = z_support.contiguous()

        z_proto = z_support.view(self.n_way, self.n_support, -1).mean(1) # the shape of z is [n_data,n_dim]

        z_query = z_query.contiguous().view(self.n_way*self.n_query,-1)

        dists = eulidean_distance(z_query, z_proto)
        scores = -dists
        return scores


    def get_distance(self, x):
        pass

    def set_forward_loss(self, x):
        y = torch.from_numpy(np.repeat(range(self.n_way), self.n_query)) #generate n_way*n_query data eg.[0,0,0,1,1,1,2,2,2,3,3,3,4,4,4] #n_way=5,n_support=3
        if self.use_cuda:
            y = y.cuda()
        scores = self.set_forward(x)
        loss = self.loss_fn(scores, y)

        return scores, loss

def eulidean_distance(x, y):
    # x:NxD
    # y:MxD
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n,m,d)
    y = y.unsqueeze(0).expand(n,m,d)

    return torch.pow(x-y, 2).sum(2)


def proto_test():
    x = torch.randn((5, 6, 3, 84, 84)) #n_way=5, n_support+n_query=6
    # feature_model = backbone.Conv4NP()
    model_w_fc = models.resnet18(pretrained=False)
    seq = list(model_w_fc.children())[:-1]
    feature_model = nn.Sequential(*seq)

    feature_model.final_feat_dim = [512,1,1]
    model = ProtoNet(feature_model, n_way=5, n_support=3, n_query=3)
    # model.n_query = int(x.size(0)/model.n_way) - model.n_support #TODO

    # model.parse_feature(x, is_feature=False)
    scors,loss = model.set_forward_loss(x)
    print(scors.shape)

if __name__ == '__main__':
    proto_test()
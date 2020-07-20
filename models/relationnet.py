import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.meta_template import MetaTemplate
from models import backbone
import utils


class RelationNet(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, n_query, tf_path=None,loss_type='mse', use_cuda = True):
        super(RelationNet, self).__init__(model_func, n_way, n_support, n_query, use_cuda=use_cuda)

        # loss function
        self.loss_type = loss_type  # mse or softmax
        if self.loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif self.loss_type == 'softmax':
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            print("loss type only support mse or softmax")

        # metric function
        self.relation_module = RelationModule(self.feat_dim, 8, self.loss_type)
        self.method = 'RelationNet'

    def set_forward(self, x, is_feature=False):
        # print("set_forward x.data.device:",x.data.device)
        #get features
        z_support, z_query = self.parse_feature(x, is_feature)
        z_support = z_support.contiguous()
        # print(z_support.shape)
        # z_proto.shape: (n_way, c,w,h)
        z_proto = z_support.view(self.n_way, self.n_support, *self.feat_dim).mean(1) #the proto of class i used by mean
        # z_query.shape: (n_way*n_query, c,w,h)
        z_query = z_query.contiguous().view(self.n_way*self.n_query, *self.feat_dim)

        #get relations with metric function
        z_proto_ext = z_proto.unsqueeze(0).repeat(self.n_query*self.n_way, 1,1,1,1)
        # print(z_proto_ext.shape)
        z_query_ext = z_query.unsqueeze(0).repeat(self.n_way,1,1,1,1)
        z_query_ext = torch.transpose(z_query_ext,0,1)
        # print(z_query_ext.shape)

        extend_final_feat_dim = self.feat_dim.copy()
        extend_final_feat_dim[0]*=2
        replation_pairs = torch.cat((z_proto_ext,z_query_ext),2).view(-1, *extend_final_feat_dim)
        relations = self.relation_module(replation_pairs).view(-1, self.n_way)
        return relations



    def set_forward_loss(self, x):
        y = torch.from_numpy(np.repeat(range(self.n_way), self.n_query)) #generate n_way*n_query data eg.[0,0,0,1,1,1,2,2,2,3,3,3,4,4,4] #n_way=5,n_support=3

        scores = self.set_forward(x)
        if self.loss_type == 'mse':
            y_oh = utils.one_hot(y, self.n_way)
            if self.use_cuda:
                y_oh = y_oh.cuda()
            loss = self.loss_fn(scores, y_oh)
        else: #softmax
            if self.use_cuda:
                y = y.cuda()
            loss = self.loss_fn(scores, y)
        return scores, loss



# --- Simple Conv Block ---
class RelationConvBlock(nn.Module):
    maml = False

    def __init__(self, indim, outdim, padding=0):
        super(RelationConvBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            pass  # TODO
        else:
            self.C = nn.Conv2d(indim, outdim, 3, padding=padding)
            self.BN = nn.BatchNorm2d(outdim, momentum=1, affine=True, track_running_stats=False)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

        self.parameterized_layers = [self.C, self.BN, self.relu, self.pool]
        for layer in self.parameterized_layers:
            backbone.init_layer(layer)

        self.trunk = nn.Sequential(*self.parameterized_layers)

    def forward(self, x):
        out = self.trunk(x)
        return out


# --- Relation module adopted in RelationNet ---
class RelationModule(nn.Module):
    maml = False

    def __init__(self, input_size, hidden_size, loss_type='mse'):
        super(RelationModule, self).__init__()

        self.loss_type = loss_type
        padding = 1 if (input_size[1] < 10) and (input_size[2] < 10) else 0

        self.layer1 = RelationConvBlock(input_size[0] *2, input_size[0], padding=padding)
        self.layer2 = RelationConvBlock(input_size[0], input_size[0], padding=padding)

        shrink_s = lambda s: int((int((s - 2 + 2 * padding) / 2) - 2 + 2 * padding) / 2)

        if self.maml:
            pass  # TODO
        else:
            self.fc1 = nn.Linear(input_size[0] * shrink_s(input_size[1]) * shrink_s(input_size[2]), hidden_size)#in relationnet paper, this is nn.Linear(64*3*3 ,8), here is also nn.Linear(64*3*3 ,8) after shrink_s
            self.fc2 = nn.Linear(hidden_size, 1)



    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        if self.loss_type == 'mse':
            out = torch.sigmoid(self.fc2(out))
        elif self.loss_type == 'softmax':
            out = self.fc2(out)
        return out


def main():
    x = torch.randn((30, 3, 84, 84))
    feature_model = backbone.Conv4NP()
    out = feature_model(x)
    print(out.dim())
    print("after feature:",out.shape)
    # model = RelationNet(feature_model, loss_type='mse', n_way=5, n_support=5)
    # model = RelationModule(64, 8)
    # out = model(x)
    # print(out.shape)

    relation = RelationModule([64,19,19], 8)
    out = relation(out)
    print(out.shape)


def relationnet_test():
    x = torch.randn((5, 6, 3, 84, 84)) #n_way=5, n_support+n_query=6
    feature_model = backbone.Conv4NP()
    model = RelationNet(feature_model, n_way=5, n_support=3, n_query=3)
    # model.n_query = int(x.size(0)/model.n_way) - model.n_support #TODO

    # model.parse_feature(x, is_feature=False)
    scors,loss = model.set_forward_loss(x)
    print(scors.shape)

    # out = model(x)
    # print(out.shape)


def mse_loss_test():
    x = torch.Tensor([[1,2,3],[0,2,3]])
    y = torch.Tensor([[0,0,0],[0,0,0]])
    loss = nn.MSELoss()
    l = loss(x,y)
    print(l)


if __name__ == '__main__':
    relationnet_test()
    # mse_loss_test()
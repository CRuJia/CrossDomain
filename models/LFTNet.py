import torch
import torch.nn as nn

from models import protonet
from models import relationnet
from models import backbone
from tensorboardX import SummaryWriter


class LFTNet(nn.Module):
    def __init__(self, args, tf_path=None, change_way=True):
        super(LFTNet, self).__init__()

        #tf writer
        self.tf_writer = SummaryWriter(log_dir=tf_path) if tf_path is not None else None

        train_few_shot_params = dict(n_way=args.n_way, n_support=args.n_support, n_query=args.n_query)
        if args.method == 'protonet':
            pass #TODO
        elif args.method == 'matchingnet':
            pass #TODO
        elif args.method == 'relationnet':
            if args.model == 'Conv4':
                feature_model = backbone.Conv4NP()
            elif args.model == 'Conv6':
                feature_model = backbone.Conv6NP()
            else:  # TODO resnet
                feature_model = backbone.Conv6NP()
                pass
            loss_type = 'mse' if args.method == 'relationnet' else 'softmax'
            model = relationnet.RelationNet(feature_model, loss_type=loss_type, tf_path=args.tf_dir, use_cuda=args.use_cuda, **train_few_shot_params)
        elif args.method == 'gnn':
            pass #TODO gnn
        else:
            raise ValueError('Unknown method')
        self.model = model
        print(" train withg {} framework".format(args.method))

        # for auxiliary training
        #TODO

        #optimizet
        model_params, ft_params = self.split_model_parameters()
        self.model_optim = torch.optim.Adam(model_params) #TODO add auxiliary parameters
        self.ft_optim = torch.optim.Adam(ft_params, weight_decay=1e-8, lr=1e-3)

        #total epochs
        self.total_epoch = args.end_epoch


    # split the parameters of feature-wise tansformation layers and others
    def split_model_parameters(self):
        model_params = []
        ft_params = []
        for n,p in self.model.named_parameters():
            n = n.splot('.')
            if n[-1] == 'gamma' or n[-1] == 'beta':
                ft_params.append(p)
            else:
                model_params.append(p)
        return model_params, ft_params


    def trainall_loop(self, epoch, ps_loader, pu_loader, aux_iter, total_it=None):
        #TODO
        print_freq = len(ps_loader)/10
        agv_model_loss = 0.
        afg_ft_loss = 0.

        for i, ((x,_), (x_new,_)) in enumerate(zip(ps_loader, pu_loader)):
            pass


    #train the model itself (with ft layers)
    def train_loop(self, epoch, base_loader, total_it):
        print_freq = len(base_loader)/10
        avg_model_loss = 0.

        # clear fast weight and enable ft layer
        # for weight in self.model.parameters():
        #     weight.fast = None

        #training loop
        for i, (x,_) in enumerate(base_loader):
            _, model_loss = self.model.set_forward_loss(x)


            #optimizer
            self.model_optim.zero_grad()
            model_loss.backward()
            self.model_optim.step()

            #loss
            avg_model_loss += model_loss.item()
            if (i+1)%print_freq == 0:
                print('Epoch {:d}/{:d} | Batch {:d}{:d} | model_loss {:}'.format(epoch+1, self.total_epoch, i+1, len(base_loader), avg_model_loss))
            if (i+1)% 10 == 0 and self.tf_writer is not None:
                self.tf_writer.add_scalar('LFTNet/model_loss',model_loss.item(), total_it+1)
            total_it +=1
        return total_it


    def test_loop(self, test_loader, record=False):
        self.model.eval()
        return self.model.test_loop(test_loader, record)


    #save function
    def save(self, filename, epoch):
        state = {'epoch':epoch,
                 'model_state':self.model.state_dict(),
                 'model_optim_state':self.model_optim.state_dict(),
                 'ft_optim_state':self.ft_optim.state_dict()}
        torch.save(state, filename)

    # load function
    def resume(self, filename):
        state = torch.load(filename)
        self.model.load_state_dict(state['model_state'])
        self.model_optim.load_state_dict(state['model_optim_state'])
        self.ft_optim.load_state_dict(state['ft_optim_state'])
        return state['epoch']+1



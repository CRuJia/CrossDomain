import torch
import numpy as np
from options import parse_args
from models import backbone
from models.relationnet import RelationNet
from data.datamanager import SetDataManager
from torchvision import models
import torch.nn as nn

def train(base_data_manager, val_loader, model, optimizer, start_epoch, stop_epoch, args,use_cuda=True):
    """

    :param base_data_manager:
    :param val_loader:
    :param model:
    :param start_epoch:
    :param stop_epoch:
    :param args:
    :return:
    """

    #for validation
    max_acc = 0
    total_it = 0

    #training
    for epoch in range(start_epoch,stop_epoch):
        data_loader = base_data_manager.get_data_loader(aug=False)

        # train loop
        model.train()

        pre_freq = len(data_loader)/10
        avg_model_loss = 0.

        for i, (x,_) in enumerate(data_loader):
            if use_cuda:
                x = x.cuda()
            _,loss = model.set_forward_loss(x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_model_loss +=loss.item()

            if (i+1)%pre_freq == 0:
                print('Epoch:{:d} / {:d} | Batch {:d}/{:d} | Model_loss:{:.4f}'.format(epoch+1, stop_epoch, i+1, len(data_loader), avg_model_loss/float(i+1)))


        # validate
        model.eval()
        with torch.no_grad():
            acc = model.test_loop(val_loader)
            print('Epoch:{:d} / {:d} | ACC {:.4f}%'.format(epoch+1, stop_epoch,acc))

        #save
        if acc >max_acc:
            print("best model! save...")
            max_acc = acc
            torch.save(model.state_dict(),'./resnet18_best_model.pth')






def chose_label_file(dataset):
    """

    :param dataset:  type:list or str
    :return:
    """
    #TODO
    dic = {'cars':'cars/all.csv','miniImagenet':'miniImageNet/all.csv','cub':'CUB_200_2011/all.csv','places':'places365_standard/all.csv'}
    assert isinstance(dataset,(list,str))
    if isinstance(dataset, list):
        return [dic[i] for i in dataset]
    else: #str
        return dic[dataset]





def main():
    args = parse_args('train')

    print(args)

    #set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    #dataloader
    print("\n -- prepare dataloader ---")
    train_files = chose_label_file(args.trainset)
    val_file = chose_label_file(args.testset)

    if 'Conv' in args.model:
        image_size = 84
    else:
        image_size = 224

    image_size = 224

    train_few_shot_params = dict(n_way=args.train_n_way, n_support=args.n_support, n_query=args.n_query, n_episode=args.n_episode)
    base_data_manager = SetDataManager(image_size=image_size, **train_few_shot_params, data_file=train_files)

    test_few_shot_params = dict(n_way=args.test_n_way, n_support=args.n_support, n_query=args.n_query, n_episode=args.n_episode)
    val_data_manager = SetDataManager(image_size=image_size, **test_few_shot_params, data_file=val_file)
    val_loader = val_data_manager.get_data_loader()

    # model
    print("--- load model ---")
    # feature_model = backbone.Conv4NP()

    model_w_fc = models.resnet18(pretrained=False)
    seq = list(model_w_fc.children())[:-2]
    feature_model = nn.Sequential(*seq)

    feature_model.final_feat_dim = [512, 7, 7]

    model = RelationNet(feature_model, n_way=args.train_n_way, n_support=args.n_support, n_query = args.n_query, use_cuda=args.use_cuda)

    if args.use_cuda:
        model.cuda()
        print("model load to GPU")

    start_epoch = args.start_epoch
    end_epoch = args.end_epoch

    optimizer = torch.optim.Adam(model.parameters())
    train(base_data_manager,val_loader,model, optimizer, start_epoch=start_epoch, stop_epoch= end_epoch, args=args)





if __name__ == '__main__':
    main()
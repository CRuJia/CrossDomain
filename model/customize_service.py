
#!/usr/bin/python
# -*- coding: UTF-8 -*-


from __future__ import absolute_import

from __future__ import division

from __future__ import print_function


from models import backbone
from models.relationnet import RelationNet

import os
from math import exp
import numpy as np

from PIL import Image
import cv2
from model_service.pytorch_model_service import PTServingBaseService
import torch.nn as nn
import torch
torch.manual_seed(0)
import logging
import torchvision.models as models
import torchvision.transforms as transforms


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


infer_transformation = transforms.Compose([
    transforms.Resize((84, 84)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


IMAGES_KEY = 'images'
MODEL_INPUT_KEY = 'images'


def decode_image(file_content):

    """
    Decode bytes to a single image
    :param file_content: bytes
    :return: ndarray with rank=3
    """

    image = Image.open(file_content)
    image = image.convert('RGB')
    # print(image.shape)
   # image = np.asarray(image, dtype=np.float32)
    return image
#    image_content = r.files[file_content].read() # python 'List' class that holds byte
#    np_array = np.fromstring(image_content, np.uint8) # numpy array with dtype np.unit8
#    img_array = cv2.imdecode(np_array, cv2.IMREAD_COLOR) # numpy array in shape [height, width, channels]
 

class CrossDomainService(PTServingBaseService):
    def __init__(self, model_name, model_path):

        super(CrossDomainService, self).__init__(model_name, model_path)
        # self.model = resnet18(model_path)
        self.model = relationnet(model_path)
        dir_path = os.path.dirname(os.path.realpath(self.model_path))

    def _preprocess(self, data):

        """
        `data` is provided by Upredict service according to the input data. Which is like:
          {
              'images': {
                'image_a.jpg': b'xxx'
              }
          }
        For now, predict a single image at a time.
        """
        preprocessed_data = {}
        input_batch = []
        for file_name, file_content in data[IMAGES_KEY].items():
            
            print('\tAppending image: %s' % file_name)

            image1 = decode_image(file_content)
            if torch.cuda.is_available():
                input_batch.append(infer_transformation(image1).cuda())
            else:
                input_batch.append(infer_transformation(image1))
        input_batch_var = torch.autograd.Variable(torch.stack(input_batch, dim=0), volatile=True)
        preprocessed_data[MODEL_INPUT_KEY] = input_batch_var

       # print('preprocessed_data',input_batch_var.shape())

        return preprocessed_data

    def _postprocess(self, data):

        """
        `data` is the result of your model. Which is like:
          {
            'feature': [[0.1, -0.05, ..., 0.34]]
          }
        value of feature is a single list of list because one image is predicted at a time for now.
        """


        feature =  data['images'][0].detach().numpy().tolist()

        output = {'feature': feature}

        return output

def relationnet(model_path, **kwargs):
    feature_model = backbone.Conv4NP()
    model = RelationNet(feature_model, n_way=5, n_support=5, n_query=5)

    seq = list(model.feature.children())
    seq.append(Flatten())
    model = torch.nn.Sequential(*seq)

    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    model.eval()


    return model


def resnet18(model_path, **kwargs):

    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model_w_fc = models.resnet18(pretrained=False)
    seq = list(model_w_fc.children())[:-1]
    seq.append(Flatten())
    model = torch.nn.Sequential(*seq)
    # model.load_state_dict(torch.load(model_path), strict=False)
    model.load_state_dict(torch.load(model_path, map_location ='cpu'), strict=False)
    # model.load_state_dict(torch.load(model_path))

    model.eval()

    return model

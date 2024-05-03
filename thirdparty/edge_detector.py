import kornia as kn
import argparse
from thirdparty.pidinet import models
import torch
from types import SimpleNamespace

def get_edge(img):
    #img [1,3,512,512]
    filter = kn.filters.Sobel()
    img_edge = filter(img)

    return img_edge #[1,1,512,512]

'''PidiNet egge detector'''
# parser = argparse.ArgumentParser(description='PyTorch Pixel Difference Convolutional Networks')
# parser.add_argument('--pidinet_model', type=str, default='pidinet_tiny',
#         help='model to train the dataset')
# parser.add_argument('--sa', action='store_false',
#         help='use CSAM in pidinet')
# parser.add_argument('--dil', action='store_false',
#         help='use CDCM in pidinet')
# parser.add_argument('--config', type=str, default='carv4',
#         help='model configurations, please refer to models/config.py for possible configurations')
# parser.add_argument('--evaluate', type=str, default="thirdparty/pidinet/trained_models/table5_pidinet-tiny.pth",
#         help='full path to checkpoint to be evaluated')
#
# parser.add_argument('--gpu', type=str, default='0', help='gpus available')
# pidnet_args = parser.parse_args()

# create a namespace object instead of using parser.parse_args()
pidnet_args = SimpleNamespace(config='carv4', dil=True, evaluate='thirdparty/pidinet/trained_models/table5_pidinet-tiny.pth', gpu='0', pidinet_model='pidinet_tiny', sa=True)


def Initialize_PidNet(args):
    ### Create model
    model = getattr(models, args.pidinet_model)(args)

    ###Transfor to cuda devices
    model = torch.nn.DataParallel(model).cuda()

    ###load model
    state = torch.load(args.evaluate, map_location='cpu')
    model.load_state_dict(state['state_dict'])
    return model

def PidNet(model,img):
    _,_,H,W = img.shape
    with torch.no_grad():
        results = model(img)
        result = results[-1]
    return result
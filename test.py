import cv2
import os
import argparse
import glob
from models.RDN import *

from utils import *

import time
import cv2
from PIL import Image
from einops import rearrange

import scipy.io as scio
import torchvision.transforms as transforms

from functools import reduce


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--which_model", type=str, default='final_net.pth', help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs/", help='path of log files')
parser.add_argument("--name", type=str, default=r"D:\Denoising-master\Denoising-master\logs\float16\28_net.pth", help='model name')
parser.add_argument("--test_path", type=str, default="data/BenchmarkNoisyBlocksSrgb.mat", help='path of val files')

parser.add_argument("--add_BN", type=bool, default=True, help='Batch Normalization')


opt = parser.parse_args()

def normalize(data):
    return data/255.

def main():
    # Build model
    print('Loading model ...\n')
   
    model = RDN(64, 3)

    device_ids = [0]
    # model = nn.DataParallel(net, device_ids=device_ids).cuda()
    # original saved file with DataParallel
    state_dict = torch.load(opt.name) # 模型可以保存为pth文件，也可以为pt文件。
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
        new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。
    # load params
    model.load_state_dict(new_state_dict)  # 从新加载这个模型。
    # model.load_state_dict(torch.load(opt.name))
    model.eval()
    # load data info
    print('Loading data info ...\n')
    # files_source = glob.glob(os.path.join('data', opt.test_data, '*.png'))
    # files_source.sort()
    # mat_file = scio.loadmat(opt.test_path)
    # data = mat_file['BenchmarkNoisyBlocksSrgb']

    # process data
    ave_time = 0
    cnt = 1
    for p in range(801,901):
            # image
        n_img = Image.open(r"D:\noise\noise\0{}.png".format(p))


        # Convert to numpy
        data = np.array(n_img, dtype=np.float16)
        h=data.shape[1]
        img = np.float32(normalize(data))

        input = transforms.ToTensor()(img)
        input=input.transpose(2,1)
        input = input.unsqueeze(0)

        # input = input.cuda()

        with torch.no_grad():  # this can save much memory
            torch.cuda.synchronize()
            start = time.time()
            out = model(input)
            torch.cuda.synchronize()
            end = time.time()
            ave_time = ave_time + end - start

            out = torch.clamp(out, 0., 1.) * 255
            out_img = out.squeeze(0).cpu().numpy()
            out_img = out_img.astype('uint8')
            out_img = np.transpose(out_img, (1, 2, 0))

            cnt = cnt + 1
                # print(cnt)

            out_img = cv2.flip(out_img, 1)
            out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)

            out_img = cv2.rotate( out_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(r"D:\DIV2K_valid_HR\result\0{}.png".format(p), out_img)




    model_dir = os.path.join('data', 'Resultstest')
    print('create checkpoint directory %s...' % model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)



    ave_time = ave_time / (1280)
    ave_time = ave_time * (1000/256) * (1000/256)
    print('average time : %4f', ave_time)

if __name__ == "__main__":
    main()

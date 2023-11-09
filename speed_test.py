import argparse
from torch.utils.data import DataLoader
import utils
import os
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import time

### Dataloader ###
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])
def get_test_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, img_options)
class DataLoaderTest(Dataset):
    def __init__(self, inp_dir, img_options):
        super(DataLoaderTest, self).__init__()
        inp_files = sorted(os.listdir(inp_dir))
        self.inp_filenames = [os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)]
        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):
        path_inp = self.inp_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]
        inp = Image.open(path_inp)
        inp = TF.to_tensor(inp)
        return inp, filename

if __name__=="__main__":
    ## parameter ##
    parser = argparse.ArgumentParser(description='Image Deblurring')
    ## 输入文件夹路径 ##
    parser.add_argument('--input_dir', default='./test_img', type=str, help='Directory of validation images')
    parser.add_argument('--weights', default='./checkpoints/DeepRFT/model_GoPro.pth', type=str, help='Path to weights')
    parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    ## Initialize your net, please edit it to suit for your net  ##
    ## 这里请导入需要的模型类 ##
    from DeepRFT_MIMO import DeepRFT as mynet
    model = mynet(num_res=8, inference=True)

    ## Load the checkpoints  ##
    checkpoint = torch.load(args.weights)
    model.load_state_dict(checkpoint["state_dict"])

    # utils.load_checkpoint_compress_doconv(model, args.weights) # 在deeprft中使用了do卷积，所以采用特殊加载方式。

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    ## Get dataloader ##
    data_dir = args.input_dir
    test_dataset = get_test_data(data_dir, img_options={})
    test_loader  = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)


    all_time = 0.
    ## main loop ##
    with torch.no_grad():
        for ii, data_test in enumerate(test_loader):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            input = data_test[0].cuda()

            torch.cuda.synchronize()
            start = time.time()
            ## Output from your net maybe different from my net. Type: Tensor, Size: [B,C,H,W] ##
            ## 输出可能需要根据不同的网络进行调整，请仔细看模型类 ##
            restored = model(input)
            restored = torch.clamp(restored, 0, 1)
            restored = torch.clamp(restored, 0, 1)
            # print(restored.shape)
            torch.cuda.synchronize()
            end = time.time()
            all_time += end - start
    print('average_time: ', all_time / len(test_dataset))





"""
Training code for Adversarial patch training
For classification

"""

import PIL
import load_data
from tqdm import tqdm

from load_data import *
import gc
import matplotlib.pyplot as plt
from torch import autograd
from torchvision import transforms
from tensorboardX import SummaryWriter
import subprocess
import patch_config
import sys
import time
import torchvision.models as models
import torch
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

import importlib
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

pth = '../../weights/resnet50_cifar100.pth'
data = '../../data/cls/cifar-100-python.tar.gz'
cifar_path = '../../data/cls/cifar100-pic-version/'



# assert os._exists(pth)
# 加载模型



def load_cifar100(root, train=True, transform=None):
    data_dir = os.path.join(root, 'train' if train else 'test')
    dataset = ImageFolder(data_dir, transform=transform)
    return dataset

def load_darts_model(model_file='./weights/eval-EXP-20230715-071805'):
    """加载模型

    """
    assert os.path.exists(model_file)
    file_path = os.path.join(model_file, 'scripts', 'train.py')

    # 指定模块名
    module_name = "Network"

    # 创建一个ModuleSpec对象，用于指定模块的源代码文件
    spec = importlib.util.spec_from_file_location(module_name, file_path)

    # 加载并返回该模块
    Network = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(Network)
    CIFAR_CLASSES = 10

    # 现在可以使用mypackage中的函数和变量了
    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
    model = model.cuda()
    return model


class PatchTrainer(object):
    def __init__(self, mode, args):
        self.config = patch_config.patch_configs[mode]()
        self.img_size_height = 32
        model = models.resnet50(pretrained=False)
        if args.data_type == 'cifar100':
        # self.darknet_model = Darknet(self.config.cfgfile)
        # self.darknet_model.load_weights(self.config.weightfile)
        # self.darknet_model = self.darknet_model.eval().cuda() # TODO: Why eval?
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, 100)  # 替换最后一层

        # 加载预训练权重
        model.load_state_dict(torch.load(pth))
        # model.load_state_dict(torch.jit.load(pth))
        # 设置模型为评估模式
        model.eval().cuda()
        self.darknet_model = model
        # self.cls_model = model

        self.patch_applier = PatchApplier().cuda()

        # self.patch_transformer = PatchTransformer().cuda()  # TODO:修改patch适应环境性
        self.patch_transformer_cls = PatchTransformer_cls().cuda()

        self.prob_extractor = MaxProbExtractor(0, 80, self.config).cuda()
        self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size).cuda()
        self.total_variation = TotalVariation().cuda()

        self.writer = self.init_tensorboard(mode)
        self.args = args

    def init_tensorboard(self, name=None):
        """
        BUG ：启动不了 tensorboard将 subprocess.Popen() 函数调用中的命令行参数修改为以下格式：
                subprocess.Popen(['/path/to/tensorboard', '--logdir=runs'])
            windows subprocess.Popen(['C:\\path\\to\\tensorboard.exe', '--logdir=runs'])
        """
        # subprocess.Popen([r'c:\users\lutao\appdata\roaming\python\python310\site-packages\tensorboard.exe', '--logdir=runs'])
        subprocess.Popen(['tensorboard', '--logdir=runs'], shell=True)
        # 在windows 在子进程中使用echo，需要设置 shell =True，因为 echo 不是单独的命令，而是window CMD 内置的命令
        # FileNotFoundError: [WinError 2] 系统找不到指定的文件。
        # !pip show tensorboard
        if name is not None:
            time_str = time.strftime("%Y%m%d-%H%M%S")
            return SummaryWriter(f'runs/{time_str}_{name}')
        else:
            return SummaryWriter()

    def train(self):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """

        img_size = self.img_size_height
        batch_size = self.config.batch_size
        print(batch_size, "BS")
        # batch_size = 56
        n_epochs = args.epochs
        max_lab = 14
        print('img_size',img_size,'batch_size',batch_size,'n_epochs', n_epochs, 'max_lab', max_lab)

        time_str = time.strftime("%Y%m%d-%H%M%S")

        # Generate stating point
        adv_patch_cpu = self.generate_patch("gray")
        #adv_patch_cpu = self.read_image("saved_patches/patchnew0.jpg")

        adv_patch_cpu.requires_grad_(True)

        # train_loader = torch.utils.data.DataLoader(InriaDataset(self.config.img_dir, self.config.lab_dir, max_lab, img_size,
        #                                                         shuffle=True),
        #                                            batch_size=batch_size,
        #                                            shuffle=True,
        #                                            num_workers=10)
        train_loader, _ = self.cls_dataloader()
        self.epoch_length = len(train_loader)
        print(f'One epoch is {len(train_loader)}')

        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)

        et0 = time.time()
        for epoch in range(n_epochs):
            ep_det_loss = 0
            ep_nps_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            bt0 = time.time()
            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',
                                                        total=self.epoch_length):
                with autograd.detect_anomaly():
                    img_batch = img_batch.cuda()
                    lab_batch = lab_batch.cuda()
                    # label 不同
                    #print('TRAINING EPOCH %i, BATCH %i'%(epoch, i_batch))
                    adv_patch = adv_patch_cpu.cuda()    # torch.size: 3 210 297
                    # lab_batch需要提供信息，cls和od中不同[image.size(0)]
                    # img_info (list)：形状为 (batch_size, x, y ) 的标签批处理张量。
                    adv_batch_t = self.patch_transformer_cls(adv_patch, [img_batch.size(0), img_batch.size(3),
                                            img_batch.size(2)], img_size, do_rotate=True, rand_loc=False)
                    # adv_batch_t = adv_patch
                    # 矩形：torch.size: 56 14 3 329 416   329哪儿来的 减出来的
                    # 方形：torch.size: 56 14 3 416 416 BNCHW
                    # 7.13 报错维度不匹配  adv_batch 需要在非patch地方变成o, 在前一步的变换中完成
                    p_img_batch = self.patch_applier(img_batch, adv_batch_t) # torch.Size([56, 3, 329, 416])
                    p_img_batch = F.interpolate(p_img_batch, (self.img_size_height, self.img_size_height))

                    # img = p_img_batch[1, :, :,]
                    # img = transforms.ToPILImage()(img.detach().cpu())
                    # #img.show()


                    output = self.darknet_model(p_img_batch) # 56 100
                    # max_prob = self.prob_extractor(output)
                    # TODO 异常数值，可能是模型没有加入softmax
                    max_prob, pred_label = torch.max(output, dim=1)
                    # max_prob = max_prob / torch.sum(max_prob)
                    nps = self.nps_calculator(adv_patch)
                    tv = self.total_variation(adv_patch)

                    nps_loss = nps*0.01
                    tv_loss = tv*2.5
                    det_loss = torch.mean(max_prob)*0.1
                    loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())

                    ep_det_loss += det_loss.detach().cpu().numpy()
                    ep_nps_loss += nps_loss.detach().cpu().numpy()
                    ep_tv_loss += tv_loss.detach().cpu().numpy()
                    ep_loss += loss

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    adv_patch_cpu.data.clamp_(0,1)       #keep patch in image range

                    bt1 = time.time()
                    if i_batch%5 == 0:
                        iteration = self.epoch_length * epoch + i_batch
                        self.writer.add_scalar('nps', nps.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('tv', tv.detach().cpu().numpy(), iteration)

                        self.writer.add_scalar('total_loss', loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/nps_loss', nps_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/tv_loss', tv_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('misc/epoch', epoch, iteration)
                        self.writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)

                        self.writer.add_image('patch', adv_patch_cpu, iteration)
                    if i_batch + 1 >= len(train_loader):
                        print('\n')
                    else:
                        del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss
                        torch.cuda.empty_cache()
                    bt0 = time.time()
            et1 = time.time()
            ep_det_loss = ep_det_loss/len(train_loader)
            ep_nps_loss = ep_nps_loss/len(train_loader)
            ep_tv_loss = ep_tv_loss/len(train_loader)
            ep_loss = ep_loss/len(train_loader)

            #im = transforms.ToPILImage('RGB')(adv_patch_cpu)
            #plt.imshow(im)
            #plt.savefig(f'pics/{time_str}_{self.config.patch_name}_{epoch}.png')

            scheduler.step(ep_loss)
            if True:
                print('  EPOCH NR: ', epoch),
                print('EPOCH LOSS: ', ep_loss)
                print('  DET LOSS: ', ep_det_loss)
                print('  NPS LOSS: ', ep_nps_loss)
                print('   TV LOSS: ', ep_tv_loss)
                print('EPOCH TIME: ', et1-et0)
                im = transforms.ToPILImage('RGB')(adv_patch_cpu)
                # plt.imshow(im)
                # plt.show()
                im.save("saved_patches/patchnew1.jpg")
                del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss
                torch.cuda.empty_cache()
            et0 = time.time()

    def generate_patch(self, type):
        """
        Generate a random patch as a starting point for optimization.

        :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
        :return:
        """
        if self.config.patch_name == "A4RealWorld":
            if type == 'gray':
                adv_patch_cpu = torch.full((3, self.config.patch_size_x, self.config.patch_size_y), 0.5)
            elif type == 'random':
                adv_patch_cpu = torch.rand((3, self.config.patch_size_x, self.config.patch_size_y))

        else:
            if type == 'gray':
                adv_patch_cpu = torch.full((3, self.config.patch_size, self.config.patch_size), 0.5)
            elif type == 'random':
                adv_patch_cpu = torch.rand((3, self.config.patch_size, self.config.patch_size))

        return adv_patch_cpu

    def read_image(self, path):
        """
        Read an input image to be used as a patch

        :param path: Path to the image to be read.
        :return: Returns the transformed patch as a pytorch Tensor.
        """
        patch_img = Image.open(path).convert('RGB')
        tf = transforms.Resize((self.config.patch_size, self.config.patch_size))
        patch_img = tf(patch_img)
        tf = transforms.ToTensor()

        adv_patch_cpu = tf(patch_img)
        return adv_patch_cpu

    def cls_dataloader(self):
                # 数据增强
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # 加载数据集


        # 加载训练集和测试集
        train_dataset = load_cifar100(cifar_path, train=True, transform=train_transform)
        test_dataset = load_cifar100(cifar_path, train=False, transform=test_transform)

        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.batch_size,
                                                   shuffle=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.config.batch_size,
                                                  shuffle=False, num_workers=4)

        return (train_loader,test_loader)




def main(args):
    # if len(sys.argv) != 2:
    #     print('You need to supply (only) a configuration mode.')
    #     print('Possible modes are:')
    #     print(patch_config.patch_configs)


    trainer = PatchTrainer('paper_obj', args )
    trainer.train()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("cifar")
    parser.add_argument('--set', type=str, default='cifar10', help='location of the data corpus')
    parser.add_argument('--data_type', type=str, default='CIFAR10', help='dataset that is used')
    parser.add_argument('--data', type=str, default='../../dataset/CIFAR10', help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
    parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
    parser.add_argument('--save', type=str, default='Data-Pruning', help='experiment name')
    # interface of whole process
    parser.add_argument('--model_path_1', type=str, default='../../NAS/pc_darts/eval-EXP-20230715-071805',
                        help='Path to trained models')
    parser.add_argument('--arch', type=str, default='PCDARTS', help='which architecture to use')
    parser.add_argument('--patch_file_path', type=str, default='../../patch_file', help='Path to save patches')
    parser.add_argument('--NAS_type', type=str, default='DARTS', help='type of NAS')
    #parser.add_argument('--model_path_2', type=str, default='./weights/yolo.weights', help='Path to pretrained models')
    #parser.add_argument('--model_path_3', type=str, default='./weights/yolo.weights', help='Path to pretrained models')

    args = parser.parse_args()
    # main(args)
    load_darts_model()


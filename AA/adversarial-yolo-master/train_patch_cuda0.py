"""
Training code for Adversarial patch training


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
from tools.simlarity.sim_calculate import SimScore


class PatchTrainer(object):
    def __init__(self, mode):
        self.config = patch_config.patch_configs[mode]()

        self.darknet_model = Darknet(self.config.cfgfile)
        self.darknet_model.load_weights(self.config.weightfile)
        self.darknet_model = self.darknet_model.eval().cuda() # TODO: Why eval?
        self.patch_applier = PatchApplier().cuda()
        self.patch_transformer = PatchTransformer().cuda()  # TODO:修改patch适应环境性
        self.prob_extractor = MaxProbExtractor(0, 80, self.config).cuda()
        if self.config.patch_name in self.config.none_squ_list:
            self.nps_calculator = NPSCalculator_rect(self.config.printfile,  (self.config.patch_size_x,  self.config.patch_size_y)).cuda()
        else:
            self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size).cuda()
        self.total_variation = TotalVariation().cuda()

        self.writer = self.init_tensorboard(mode)

        self.sim_scorer = SimScore(cal_dev=1, fea_method='canny', sim_method='rmse')    # 相似度衡量
        self.gpuid = 0

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

        img_size = self.darknet_model.height
        batch_size = self.config.batch_size
        print(batch_size, "BS")
        # batch_size = 56
        n_epochs = 5000
        max_lab = 14
        print('img_size',img_size,'batch_size',batch_size,'n_epochs', n_epochs, 'max_lab', max_lab)

        time_str = time.strftime("%Y%m%d-%H%M%S")

        # Generate stating point
        adv_patch_cpu = self.generate_patch("gray")
        #adv_patch_cpu = self.read_image("saved_patches/patchnew0.jpg")

        adv_patch_cpu.requires_grad_(True)

        train_loader = torch.utils.data.DataLoader(InriaDataset(self.config.img_dir, self.config.lab_dir, max_lab, img_size,
                                                                shuffle=True),
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=10)
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
                    #print('TRAINING EPOCH %i, BATCH %i'%(epoch, i_batch))
                    adv_patch = adv_patch_cpu.cuda()    # adv_patch.size: 3 210 297
                    # x = torch.zeros(3, 210, 210)
                    # x = x.cuda()
                    # adv_patch.size: 3 210 297   img_size 416
                    adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=False)
                    # 矩形：torch.size: 56 14 3 329 416   329哪儿来的 减出来的
                    # 方形：torch.size: 56 14 3 416 416 BNCHW
                    # 更改A4后，这边报错 # Done
                    p_img_batch = self.patch_applier(img_batch, adv_batch_t)    # torch.Size([56, 3, 329, 416])
                    p_img_batch = F.interpolate(p_img_batch, (self.darknet_model.height, self.darknet_model.width))

                    # img = p_img_batch[1, :, :,]
                    # img = transforms.ToPILImage()(img.detach().cpu())
                    # img.show()


                    output = self.darknet_model(p_img_batch)
                    max_prob = self.prob_extractor(output)
                    nps = self.nps_calculator(adv_patch)
                    tv = self.total_variation(adv_patch)


                    nps_loss = nps*0.01
                    tv_loss = tv*2.5
                    det_loss = torch.mean(max_prob)
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

            im = transforms.ToPILImage('RGB')(adv_patch_cpu)
            plt.imshow(im)
            folder_path = 'pics'  # 文件夹路径
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            plt.savefig(f'pics/{time_str}_{self.config.patch_name}_{epoch}.png')

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
        if self.config.patch_name in self.config.none_squ_list:
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


def main():
    if len(sys.argv) != 2:
        print('You need to supply (only) a configuration mode.')
        print('Possible modes are:')
        print(patch_config.patch_configs)


    trainer = PatchTrainer(sys.argv[1])
    trainer.train()

if __name__ == '__main__':
    main()


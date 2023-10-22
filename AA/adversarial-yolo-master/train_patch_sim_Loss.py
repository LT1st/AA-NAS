"""
Training code for Adversarial patch training


"""

import PIL
import load_data
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity
from load_data import *
import gc
import matplotlib.pyplot as plt
from torch import autograd
from torchvision import transforms
from tensorboardX import SummaryWriter
import subprocess
import torch
import torch.nn.functional as F
import torchvision.models as models

import patch_config
import sys
from torch import optim
import time
from tools.simlarity.sim_calculate import SimScore
import tools.simlarity.pytorch_ssim as pytorch_ssim


class PatchTrainer(object):
    def __init__(self, mode):
        self.config = patch_config.patch_configs[mode]()

        self.darknet_model = Darknet(self.config.cfgfile)
        self.darknet_model.load_weights(self.config.weightfile)
        self.darknet_model = self.darknet_model.eval().cuda()  # TODO: Why eval?
        self.patch_applier = PatchApplier().cuda()
        self.patch_transformer = PatchTransformer().cuda()  # TODO:修改patch适应环境性
        self.prob_extractor = MaxProbExtractor(0, 80, self.config).cuda()
        if self.config.patch_name in self.config.none_squ_list:
            self.nps_calculator = NPSCalculator_rect(self.config.printfile,
                                                     (self.config.patch_size_x, self.config.patch_size_y)).cuda()
        else:
            self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size).cuda()
        self.total_variation = TotalVariation().cuda()

        self.writer = self.init_tensorboard(mode)

        self.sim_scorer = SimScore(cal_dev=1, fea_method='canny', sim_method='ssim')  # 相似度衡量
        self.gpuid = 0
        self.device = torch.device("cuda:0")
        self.extract_model = self.get_pretrained()

    def get_pretrained(self):
        model = models.resnet18(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-4])
        model.eval()
        model = model.to(self.device)

        return model

    def extract_features(self, this_tensor, size=(28, 28)):
        this_tensor = this_tensor.to(self.device)
        # with torch.no_grad():
        features = self.extract_model(this_tensor)

        features = F.interpolate(features, size=size, mode='bilinear', align_corners=False)
        return features

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

    def visualize_tensor(self, tensor, index=0):
        # 将 Tensor 复制到 CPU
        tensor_cpu = tensor.cpu()

        # 获取指定索引位置的 CHW 维度的图像
        image_tensor = tensor_cpu[index]

        # 将 Tensor 转换为 NumPy 数组
        image_array = image_tensor.detach().numpy()

        # 可视化图像
        plt.imshow(image_array)
        plt.axis('off')
        plt.show()

    def train(self):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """
        SIM_LOSS_FLAG = False
        TIME_DEBUG = True

        img_size = self.darknet_model.height
        batch_size = self.config.batch_size
        print(batch_size, "BS")
        # batch_size = 56
        n_epochs = 5000
        max_lab = 14
        print('img_size', img_size, 'batch_size', batch_size, 'n_epochs', n_epochs, 'max_lab', max_lab)

        time_str = time.strftime("%Y%m%d-%H%M%S")

        # Generate stating point
        adv_patch_cpu = self.generate_patch("gray")
        # adv_patch_cpu = self.read_image("saved_patches/patchnew0.jpg")

        adv_patch_cpu.requires_grad_(True)

        train_loader = torch.utils.data.DataLoader(
            InriaDataset(self.config.img_dir, self.config.lab_dir, max_lab, img_size,
                         shuffle=True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=10)
        self.epoch_length = len(train_loader)
        print(f'Total epoch number in one iteration : {len(train_loader)}')

        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)

        et0 = time.time()

        for epoch in range(n_epochs):
            ep_det_loss = 0
            ep_nps_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            ep_sim_loss = 0
            ep_sim_loss_v2 = 0
            bt0 = time.time()
            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',
                                                        total=self.epoch_length):
                if TIME_DEBUG:
                    print(time.time() - bt0)
                with autograd.detect_anomaly():
                    if TIME_DEBUG:
                        print(time.time() - bt0)
                    img_batch = img_batch.cuda()
                    lab_batch = lab_batch.cuda()
                    # print('TRAINING EPOCH %i, BATCH %i'%(epoch, i_batch))
                    adv_patch = adv_patch_cpu.cuda()  # adv_patch.size: 3 210 297
                    if TIME_DEBUG:
                        print('DATA to CUDA ',time.time() - bt0)
                    # x = torch.zeros(3, 210, 210)
                    # x = x.cuda()
                    # adv_patch.size: 3 210 297   img_size 416
                    adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=False)
                    if TIME_DEBUG:
                        print('patch_transformer',time.time() - bt0)
                    p_img_batch = self.patch_applier(img_batch, adv_batch_t)  # torch.Size([56, 3, 329, 416])
                    if TIME_DEBUG:
                        print('patch_applier',time.time() - bt0)
                    p_img_batch = F.interpolate(p_img_batch, (self.darknet_model.height, self.darknet_model.width))
                    if TIME_DEBUG:
                        print('patch_applier',time.time() - bt0)

                    if SIM_LOSS_FLAG:
                        tensor1 = img_batch.to('cuda:1')
                        tensor2 = p_img_batch.to('cuda:1')
                        # print(p_img_batch.shape, "img_batch ")
                        # original_shape = tensor2.shape
                        # # 获取 B 和 N 的乘积，作为新的批次维度
                        # new_batch_size = original_shape[0] * original_shape[1]
                        # # 保持通道、高度和宽度维度不变
                        # new_shape = (new_batch_size, original_shape[2], original_shape[3], original_shape[4])
                        # # 使用 view() 方法重新调整形状
                        # output_tensor2 = tensor2.view(new_shape)
                        # output_tensor2 = output_tensor2.squeeze()
                        # print(output_tensor2.shape, "transformed output_tensor2")
                        # # 矩形：torch.size: 56 14 3 329 416   329哪儿来的 减出来的
                        # # 方形：torch.size: 56 14 3 416 416 BNCHW
                        # print("given:",adv_batch_t.shape, adv_batch_t.shape)
                        sim_score = self.sim_scorer.get_sim_score(tensor1, tensor2)
                    # print(sim_score.shape)
                    # # 更改A4后，这边报错 # Done

                    # img = p_img_batch[1, :, :,]
                    # img = transforms.ToPILImage()(img.detach().cpu())
                    # img.show()
                    img_batch_ = self.extract_features(img_batch)
                    p_img_batch_ = self.extract_features(p_img_batch)
                    if TIME_DEBUG:
                        print('extract_features',time.time() - bt0)
                    # -------------------------- 测试时间 --------------------------
                    # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                    #     with record_function("model_inference"):
                    #         sim_score_v2 = pytorch_ssim.ssim(img_batch_, p_img_batch_)
                    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
                    # -------------------------- 测试时间  --------------------------
                    sim_score_v2 = pytorch_ssim.ssim(img_batch_, p_img_batch_)

                    output = self.darknet_model(p_img_batch)
                    max_prob = self.prob_extractor(output)
                    nps = self.nps_calculator(adv_patch)
                    tv = self.total_variation(adv_patch)

                    nps_loss = nps * 0.01
                    tv_loss = tv * 2.5
                    det_loss = torch.mean(max_prob)
                    if SIM_LOSS_FLAG:
                        sim_loss = torch.sum(1 - sim_score)
                        sim_loss.cuda(0)
                    sim_loss_v2 = torch.sum(1 - sim_score_v2)
                    # sim_loss.cuda()
                    if SIM_LOSS_FLAG:
                        loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda()) \
                               + sim_loss + sim_loss_v2.cuda(0)
                    else:
                        loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda()) \
                               + sim_loss_v2.cuda(0)
                    ep_det_loss += det_loss.detach().cpu().numpy()
                    ep_nps_loss += nps_loss.detach().cpu().numpy()
                    ep_tv_loss += tv_loss.detach().cpu().numpy()
                    ep_loss += loss
                    if SIM_LOSS_FLAG:
                        ep_sim_loss += sim_loss.detach().cpu().numpy()
                    ep_sim_loss_v2 += sim_loss_v2.detach().cpu().numpy()

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    adv_patch_cpu.data.clamp_(0, 1)  # keep patch in image range

                    bt1 = time.time()
                    if i_batch % 5 == 0:
                        iteration = self.epoch_length * epoch + i_batch
                        self.writer.add_scalar('nps', nps.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('tv', tv.detach().cpu().numpy(), iteration)

                        self.writer.add_scalar('total_loss', loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/nps_loss', nps_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/tv_loss', tv_loss.detach().cpu().numpy(), iteration)
                        if SIM_LOSS_FLAG:
                            self.writer.add_scalar('loss/sim_loss', sim_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/sim_loss_v2', sim_loss_v2.detach().cpu().numpy(), iteration)
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
            ep_det_loss = ep_det_loss / len(train_loader)
            ep_nps_loss = ep_nps_loss / len(train_loader)
            ep_tv_loss = ep_tv_loss / len(train_loader)
            ep_loss = ep_loss / len(train_loader)

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
                if SIM_LOSS_FLAG:
                    print('  SIM LOSS: ', ep_sim_loss)
                print(' SIM2 LOSS: ', ep_sim_loss_v2)
                print('EPOCH TIME: ', et1 - et0)
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

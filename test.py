import argparse
import subprocess
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

from utils.dataset_utils import LowLightTestDataset, DerainDehazeDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor

from net.model import AirNet


def test_LowLight(net, dataset):
    output_path = opt.output_path + 'lowlight/'
    subprocess.check_output(['mkdir', '-p', output_path])

    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = net(x_query=degrad_patch, x_key=degrad_patch)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(restored, output_path + clean_name[0] + '.png')

        print("LowLight: psnr: %.2f, ssim: %.4f" % (psnr.avg, ssim.avg))


def test_Derain_Dehaze(net, dataset, task="derain"):
    output_path = opt.output_path + task + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = net(x_query=degrad_patch, x_key=degrad_patch)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(restored, output_path + degraded_name[0] + '.png')

        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=int, default=0,
                        help='0 for lowlight, 1 for derain, 2 for dehaze, 3 for all-in-one')

    parser.add_argument('--lowlight_dir', type=str, default="test/LowLight/", help='save path of test lowlight images')
    parser.add_argument('--derain_path', type=str, default="test/Derain/", help='save path of test raining images')
    parser.add_argument('--dehaze_path', type=str, default="test/Dehaze/", help='save path of test hazy images')
    parser.add_argument('--output_path', type=str, default="output/", help='output save path')
    parser.add_argument('--ckpt_path', type=str, default="ckpt/", help='checkpoint save path')
    opt = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(opt.cuda)

    if opt.mode == 0:
        opt.batch_size = 1
        ckpt_path = opt.ckpt_path + 'LowLight.pth'
    elif opt.mode == 1:
        opt.batch_size = 1
        ckpt_path = opt.ckpt_path + 'Derain.pth'
    elif opt.mode == 2:
        opt.batch_size = 1
        ckpt_path = opt.ckpt_path + 'Dehaze.pth'
    elif opt.mode == 3:
        opt.batch_size = 8 
        ckpt_path = opt.ckpt_path + 'AllInOne/best_model.pth'

    # Initialize datasets
    lowlight_set = LowLightTestDataset(opt)
    derain_dehaze_set = DerainDehazeDataset(opt)

    # Make network
    net = AirNet(opt).cuda()
    net.eval()

    # Safely load weights
    try:
        net.load_state_dict(torch.load(ckpt_path, map_location=torch.device(opt.cuda)))
        print(f"Successfully loaded checkpoint: {ckpt_path}")
    except FileNotFoundError:
        print("Checkpoint not found!")

    if opt.mode == 0:
        print('Start testing LowLight...')
        test_LowLight(net, lowlight_set)
    elif opt.mode == 1:
        print('Start testing rain streak removal...')
        test_Derain_Dehaze(net, derain_dehaze_set, task="derain")
    elif opt.mode == 2:
        print('Start testing dehazing...')
        test_Derain_Dehaze(net, derain_dehaze_set, task="dehaze")
    elif opt.mode == 3:
        print('=== Start All-in-One Testing ===')
        print('\nStart testing LowLight...')
        test_LowLight(net, lowlight_set)

        print('\nStart testing rain streak removal...')
        test_Derain_Dehaze(net, derain_dehaze_set, task="derain")

        print('\nStart testing dehazing...')
        test_Derain_Dehaze(net, derain_dehaze_set, task="dehaze")
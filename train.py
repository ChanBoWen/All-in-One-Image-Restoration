import os
import copy
import csv
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import TrainDataset, LowLightTestDataset, DerainDehazeDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from net.model import AirNet
from option import options as opt


def validate_task(net, testloader, task_name):
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    net.eval()  # Switch to evaluation mode

    with torch.no_grad():
        for batch in testloader:
            _, degrad, clean = batch[0], batch[1].cuda(), batch[2].cuda()

            # Returns the restored image during eval
            restored = net(x_query=degrad, x_key=degrad)

            psnr, ssim, n = compute_psnr_ssim(restored, clean)
            psnr_meter.update(psnr, n)
            ssim_meter.update(ssim, n)

    net.train()  # Switch back to training mode
    return psnr_meter.avg, ssim_meter.avg


if __name__ == '__main__':
    torch.cuda.set_device(opt.cuda)
    os.makedirs(opt.ckpt_path, exist_ok=True)

    csv_file = os.path.join(opt.ckpt_path, 'training_metrics.csv')

    trainset = TrainDataset(opt)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)

    test_opt = copy.copy(opt)
    test_opt.lowlight_dir = "test/LowLight/"
    test_opt.derain_path = "test/Derain/"
    test_opt.dehaze_path = "test/Dehaze/"

    val_lowlight = LowLightTestDataset(test_opt)
    val_derain = DerainDehazeDataset(test_opt, task="derain")
    val_dehaze = DerainDehazeDataset(test_opt, task="dehaze")

    val_loader_ll = DataLoader(val_lowlight, batch_size=1, shuffle=False)
    val_loader_dr = DataLoader(val_derain, batch_size=1, shuffle=False)
    val_loader_dh = DataLoader(val_dehaze, batch_size=1, shuffle=False)

    # Network Construction
    net = AirNet(opt).cuda()
    net.train()

    # Optimizer and Loss
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    CE = nn.CrossEntropyLoss().cuda()
    l1 = nn.L1Loss().cuda()

    best_avg_psnr = 0.0
    start_epoch = 0

    resume_file = os.path.join(opt.ckpt_path, 'resume_state.pth')
    if os.path.exists(resume_file):
        print(f"\n[INFO] Found resume state at {resume_file}!")
        checkpoint = torch.load(resume_file)

        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_avg_psnr = checkpoint['best_avg_psnr']

        print(f"[INFO] Resuming training from Epoch {start_epoch + 1}...\n")
    else:
        pretrained_path = "ckpt/All.pth" 
        
        if os.path.exists(pretrained_path):
            print(f"\n[INFO] TRANSFER LEARNING ACTIVATED!")
            print(f"Loading pre-trained brain from: {pretrained_path}")
            pretrained_dict = torch.load(pretrained_path, map_location=f'cuda:{opt.cuda}')

            model_dict = net.state_dict()
            
            filtered_dict = {}
            for k, v in pretrained_dict.items():
                if k in model_dict and v.shape != model_dict[k].shape:
                    print(f"[WARNING] Skipping '{k}' due to shape mismatch: {v.shape} -> {model_dict[k].shape}")
                    continue  # Skip this weight
                filtered_dict[k] = v
            
            # 4. Load the filtered dictionary
            net.load_state_dict(filtered_dict, strict=False)
            print("[INFO] Fine-tuning starting now...\n")
        else:
            print(f"\n[WARNING] Could not find {pretrained_path}!")
            print("[INFO] Falling back to Training from Scratch...\n")
            
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Epoch', 'Contrastive_Loss', 'L1_Loss', 
                'LL_PSNR', 'LL_SSIM', 
                'DR_PSNR', 'DR_SSIM', 
                'DH_PSNR', 'DH_SSIM', 
                'Avg_PSNR', 'Avg_SSIM'
            ])

    # Start training
    for epoch in range(start_epoch, opt.epochs):
        contrast_meter = AverageMeter()
        l1_meter = AverageMeter()

        v_ll_p = v_ll_s = v_dr_p = v_dr_s = v_dh_p = v_dh_s = v_avg_p = v_avg_s = ""

        for ([clean_name, de_id], degrad_patch_1, degrad_patch_2, clean_patch_1, clean_patch_2) in tqdm(trainloader, desc=f"Epoch {epoch+1}/{opt.epochs}"):
            degrad_patch_1, degrad_patch_2 = degrad_patch_1.cuda(), degrad_patch_2.cuda()
            clean_patch_1, clean_patch_2 = clean_patch_1.cuda(), clean_patch_2.cuda()

            optimizer.zero_grad()

            if epoch < opt.epochs_encoder:
                # Encoder pre-training
                _, output, target, _ = net.E(x_query=degrad_patch_1, x_key=degrad_patch_2)
                contrast_loss = CE(output, target)
                loss = contrast_loss
                contrast_meter.update(contrast_loss.item())
            else:
                # Full model training
                restored, output, target = net(x_query=degrad_patch_1, x_key=degrad_patch_2)
                contrast_loss = CE(output, target)
                l1_loss = l1(restored, clean_patch_1)
                loss = l1_loss + 0.1 * contrast_loss

                contrast_meter.update(contrast_loss.item())
                l1_meter.update(l1_loss.item())

            # Backward
            loss.backward()
            optimizer.step()

        if epoch < opt.epochs_encoder:
            print(f'Epoch {epoch + 1} | Contrastive Loss: {contrast_meter.avg:.4f}')
        else:
            print(f'Epoch {epoch + 1} | L1 Loss: {l1_meter.avg:.4f} | Contrastive Loss: {contrast_meter.avg:.4f}')

        if epoch >= opt.epochs_encoder and (epoch + 1) % 5 == 0:
            print("\n--- Running Validation ---")
            try:
                psnr_ll, ssim_ll = validate_task(net, val_loader_ll, "LowLight")
                psnr_dr, ssim_dr = validate_task(net, val_loader_dr, "Derain")
                psnr_dh, ssim_dh = validate_task(net, val_loader_dh, "Dehaze")

                print(f"LowLight -> PSNR: {psnr_ll:.2f} dB | SSIM: {ssim_ll:.4f}")
                print(f"Derain   -> PSNR: {psnr_dr:.2f} dB | SSIM: {ssim_dr:.4f}")
                print(f"Dehaze   -> PSNR: {psnr_dh:.2f} dB | SSIM: {ssim_dh:.4f}")

                # Calculate average PSNR across all 3 tasks
                avg_psnr = (psnr_ll + psnr_dr + psnr_dh) / 3
                avg_ssim = (ssim_ll + ssim_dr + ssim_dh) / 3

                print(f"Average PSNR: {avg_psnr:.2f} dB | Average SSIM: {avg_ssim:.4f}")

                v_ll_p, v_ll_s = round(psnr_ll, 4), round(ssim_ll, 4)
                v_dr_p, v_dr_s = round(psnr_dr, 4), round(ssim_dr, 4)
                v_dh_p, v_dh_s = round(psnr_dh, 4), round(ssim_dh, 4)
                v_avg_p, v_avg_s = round(avg_psnr, 4), round(avg_ssim, 4)

                if avg_psnr > best_avg_psnr:
                    best_avg_psnr = avg_psnr
                    print(f"⭐ New Best Average PSNR ({best_avg_psnr:.2f} dB)! Saving best_model.pth")
                    torch.save(net.state_dict(), os.path.join(opt.ckpt_path, 'best_model.pth'))

            except Exception as e:
                print(f"[WARNING] Validation skipped/failed. Ensure Test folders exist. Error: {e}")

        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            l1_val = round(l1_meter.avg, 4) if epoch >= opt.epochs_encoder else ""
            writer.writerow([
                epoch + 1, round(contrast_meter.avg, 4), l1_val, 
                v_ll_p, v_ll_s, v_dr_p, v_dr_s, v_dh_p, v_dh_s, v_avg_p, v_avg_s
            ])

        # Save checkpoint
        checkpoint_dict = net.state_dict()
        torch.save(checkpoint_dict, os.path.join(opt.ckpt_path, 'latest.pth'))

        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_avg_psnr': best_avg_psnr
        }, resume_file)

        # Learning rate scheduling
        if epoch < opt.epochs_encoder:
            # Drop LR halfway through the encoder stage
            drop_step = max(1, opt.epochs_encoder // 2)
            lr = opt.lr * (0.1 ** (epoch // drop_step))
        else:
            # Drop LR halfway through the restoration stage
            rest_epochs = opt.epochs - opt.epochs_encoder
            drop_step = max(1, rest_epochs // 2)
            lr = 0.0001 * (0.5 ** ((epoch - opt.epochs_encoder) // drop_step))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
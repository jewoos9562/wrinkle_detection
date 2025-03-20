# Copyright (c) 2023,Semin Kim, AI R&D Center, lululab

import os
import numpy as np
import random
from tqdm import tqdm
import sys

import torch
from torch.autograd import Variable
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# from WeightedDeepSupervision.loss.losses import dice_loss
# from WeightedDeepSupervision.models.unet_model import UNet_texture_front_ds
# from WeightedDeepSupervision.dataset.datasets_AIIM import Dataset_Wrinkle_WDS
# from WeightedDeepSupervision.utils.utils import add_weight_decay

wd_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "WeightedDeepSupervision"))
sys.path.append(wd_path)

from models.unet_model import UNet_texture_front_ds
from loss.losses import dice_loss
from dataset.datasets_AIIM import Dataset_Wrinkle_WDS
from utils.utils import add_weight_decay

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 43
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device :', device)
    
    if torch.cuda.is_available():
        print('Current cuda device:', torch.cuda.current_device())
        print('Count of using GPUs:', torch.cuda.device_count())

    model_id = "1"
    agum_file = '1'
    num_epochs = 200
    train_batch_size = 4
    val_batch_size = 2
    learning_rate = 0.001

    # Edit your path
    path_src = '/root/skin/wrinkle/dataset/images_resized'
    path_ttr = '/root/skin/wrinkle/dataset/textures'
    path_gnd = '/root/skin/wrinkle/dataset/GT'

    list_src = os.listdir(path_src)
    random.shuffle(list_src)
    split_ratio = 0.8
    split_idx = int(len(list_src) * split_ratio)
    list_tr = list_src[:split_idx]
    list_te = list_src[split_idx:]

    train_dataset = Dataset_Wrinkle_WDS(list_src=list_tr, path_src=path_src, path_lbl=path_gnd, path_ttr=path_ttr, b_aug=True)
    val_dataset = Dataset_Wrinkle_WDS(list_src=list_te, path_src=path_src, path_lbl=path_gnd, path_ttr=path_ttr, b_aug=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=train_batch_size, shuffle=True,
                                               num_workers=4)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=val_batch_size, shuffle=False,
                                             num_workers=2)

    str_log = 'WRINKLE_WDS'
    path_save = '/root/skin/wrinkle/saved_model/%s' % str_log
    if os.path.isdir(path_save) == False:
        os.makedirs((path_save))

    model = UNet_texture_front_ds(4, 2).to(device)

    params = add_weight_decay(model, l2_value=0.0001)
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0.000001)

    writer_loss_tr = SummaryWriter(log_dir='/root/skin/wrinkle/logs/%s/loss_tr' % str_log)
    writer_loss_te = SummaryWriter(log_dir='/root/skin/wrinkle/logs/%s/loss_te' % str_log)
    writer_jsi = SummaryWriter(log_dir='logs/%s/jsi' % str_log)

    epsilon = 2.22045e-16
    jsi_max = 0
    for epoch in range(num_epochs):
        model.train()
        batch_losses = []

        for step, (imgs, label_imgs, img_ttr, img_wds_2, img_wds_3, img_wds_4) in enumerate(tqdm(train_loader)):
            imgs = Variable(imgs).to(device)
            label_imgs = Variable(label_imgs.type(torch.LongTensor)).to(device)
            img_ttr = Variable(img_ttr).to(device)

            out_1, out_2, out_3, out_4 = model(imgs, img_ttr)
            loss = dice_loss(F.softmax(out_1, dim=1).float(), F.one_hot(label_imgs, 2).permute(0, 3, 1, 2).float(), multiclass=True)
            batch_losses.append(loss.data.cpu().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = np.mean(batch_losses)
        print('epoch %d: train loss = %.08f, lr = %.08f' % (epoch, epoch_loss, optimizer.param_groups[0]['lr']))
        scheduler.step()
        writer_loss_tr.add_scalar('loss', epoch_loss, epoch)

        model.eval()
        batch_losses = []
        y_true = []
        y_score = []
        
        for step, (imgs, label_imgs, img_ttr, _, _, _) in enumerate(tqdm(val_loader)):
            with torch.no_grad():
                imgs = Variable(imgs).to(device)
                label_imgs = Variable(label_imgs.type(torch.LongTensor)).to(device)
                img_ttr = Variable(img_ttr).to(device)
                out_1, _, _, _ = model(imgs, img_ttr)
                loss = dice_loss(F.softmax(out_1, dim=1).float(), F.one_hot(label_imgs, 2).permute(0, 3, 1, 2).float(), multiclass=True)
                batch_losses.append(loss.data.cpu().numpy())

                score = torch.softmax(out_1, dim=1).cpu().numpy()
                score = score[:, 1, :, :]
                label_imgs_np = label_imgs.cpu().numpy()
                y_score.append(score)
                y_true.append(label_imgs_np)

        y_score_np = np.concatenate([s.flatten() for s in y_score]) if y_score else np.array([])
        y_pred_np = y_score_np > 0.5 if y_score_np.size > 0 else np.array([])
        y_true_np = np.concatenate([t.flatten() for t in y_true]) if y_true else np.array([])

        epoch_loss = np.mean(batch_losses)
        jsi = metrics.jaccard_score(y_true_np, y_pred_np) if y_true_np.size > 0 else 0.0            
        print('epoch %d: val loss = %.08f' % (epoch, epoch_loss))
        print('epoch %d: jsi = %.04f\n' % (epoch, jsi))
        writer_loss_te.add_scalar('loss', epoch_loss, epoch)
        writer_jsi.add_scalar('jsi', jsi, epoch)

        if jsi_max < jsi:
            jsi_max = jsi
            fns_check = '%s/model_epoch_%d_jsi_%.4f.pth' % (path_save, epoch, jsi)
            torch.save(model.state_dict(), fns_check)
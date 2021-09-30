from __future__ import print_function
import os
import paddle
import paddle.optimizer as optim
import argparse
from paddle.io import DataLoader
import paddle.distributed as dist
from data import WiderFaceDetection, detection_collate, preproc, cfg_mnet, cfg_re152
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
from models.retinaface import RetinaFace

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--training_dataset', default='./data/widerface/train/label.txt', help='Training dataset directory')
parser.add_argument('--network', default='resnet152', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
cfg = None
if args.network == "mobile0.25":
    cfg = cfg_mnet
elif args.network == "resnet152":
    cfg = cfg_re152

rgb_mean = (104, 117, 123) # bgr order
num_classes = 2
img_dim = cfg['image_size']
num_gpu = cfg['ngpu']
batch_size = cfg['batch_size']
max_epoch = cfg['epoch']
gpu_train = cfg['gpu_train']

num_workers = args.num_workers
momentum = args.momentum
weight_decay = args.weight_decay
lr = args.lr
gamma = args.gamma
training_dataset = args.training_dataset
save_folder = args.save_folder

net = RetinaFace(cfg=cfg)
# print("Printing net...")
# print(net)

optimizer = optim.Momentum(learning_rate=lr, momentum=momentum, parameters=net.parameters(), weight_decay=weight_decay)
criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)
priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))

if args.resume_net is not None:
    print('Loading resume network...')
    net_state_dict = paddle.load(args.resume_net + '.pdparams')
    opt_state_dict = paddle.load(args.resume_net + '.pdopt')
    net.set_state_dict(net_state_dict)
    optimizer.set_state_dict(opt_state_dict)

if num_gpu > 1 and gpu_train:
    dist.init_parallel_env()
    net = paddle.DataParallel(net)
else:
    net = net.cuda()


with paddle.no_grad():
    priors = priorbox.forward()
    priors = priors.cuda()


current_lr = lr
net.train()
start_epoch = 0 + args.resume_epoch
print('Loading Dataset...')
dataset = WiderFaceDetection(training_dataset, preproc(img_dim, rgb_mean))
epoch_size = math.ceil(len(dataset)/batch_size)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=detection_collate, num_workers=num_workers)
stepvalues = (cfg['decay1'], cfg['decay2'])

for epoch in range(start_epoch, max_epoch):
    if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > cfg['decay1']):
        paddle.save(net.state_dict(), save_folder + cfg['name']+ '_epoch_' + str(epoch) + '.pdparams')
        paddle.save(optimizer.state_dict(), save_folder + cfg['name']+ '_epoch_' + str(epoch) + '.pdopt')
    if epoch in stepvalues:
        current_lr = current_lr * gamma
        optimizer.set_lr(current_lr)
      
    for i, data in enumerate(loader()):
        # with paddle.no_grad():
        load_t0 = time.time()
        images = data[0].cuda()
        targets = [anno.cuda() for anno in data[1]]
        # forward
        out = net(images)
        # backprop
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
        # print(loss)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * ((max_epoch-epoch)*epoch_size-i*cfg['batch_size']))
        print('Epoch:{}/{} || Epochiter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
              .format(epoch, max_epoch, i, epoch_size, loss_l.item(), loss_c.item(), loss_landm.item(), lr, batch_time, str(datetime.timedelta(seconds=eta))))
paddle.save(net.state_dict(), save_folder + cfg['name'] + '_Final.pdparams')
paddle.save(optimizer.state_dict(), save_folder + cfg['name']+ '_epoch_' + str(epoch) + '.pdopt')

import torch
from torch.nn import Conv2d, Module, ReLU, MaxPool2d, init
import torch.nn.functional as F
import numpy as np

def compute_loss(pafs_ys, heatmaps_ys, pafs_t, heatmaps_t, ignore_mask):
    heatmap_loss_log = []
    paf_loss_log = []
    total_loss = 0

    paf_masks = ignore_mask.unsqueeze(1).repeat([1, pafs_t.shape[1], 1, 1])
    heatmap_masks = ignore_mask.unsqueeze(1).repeat([1, heatmaps_t.shape[1], 1, 1])

    # compute loss on each stage
    for pafs_y, heatmaps_y in zip(pafs_ys, heatmaps_ys):
        stage_pafs_t = pafs_t.clone()
        stage_heatmaps_t = heatmaps_t.clone()
        stage_paf_masks = paf_masks.clone()
        stage_heatmap_masks = heatmap_masks.clone()

        if pafs_y.shape != stage_pafs_t.shape:
            with torch.no_grad():
                stage_pafs_t = F.interpolate(stage_pafs_t, pafs_y.shape[2:], mode='bilinear', align_corners=True)
                stage_heatmaps_t = F.interpolate(stage_heatmaps_t, heatmaps_y.shape[2:], mode='bilinear', align_corners=True)
                stage_paf_masks = F.interpolate(stage_paf_masks, pafs_y.shape[2:]) > 0
                stage_heatmap_masks = F.interpolate(stage_heatmap_masks, heatmaps_y.shape[2:]) > 0
                
        with torch.no_grad():       
            stage_pafs_t[stage_paf_masks == 1] = pafs_y.detach()[stage_paf_masks == 1]
            stage_heatmaps_t[stage_heatmap_masks == 1] = heatmaps_y.detach()[stage_heatmap_masks == 1]        
        
        pafs_loss = mean_square_error(pafs_y, stage_pafs_t)
        heatmaps_loss = mean_square_error(heatmaps_y, stage_heatmaps_t)

        total_loss += pafs_loss + heatmaps_loss

        paf_loss_log.append(pafs_loss.item())
        heatmap_loss_log.append(heatmaps_loss.item())

    return total_loss, np.array(paf_loss_log), np.array(heatmap_loss_log)

def mean_square_error(pred, target):
    assert pred.shape == target.shape, 'x and y should in same shape'
    return torch.sum((pred - target) ** 2) / target.nelement()

class CocoPoseNet(Module):
    insize = 368
    def __init__(self, path = None):
        super(CocoPoseNet, self).__init__()
        self.base = Base_model()
        self.stage_1 = Stage_1()
        self.stage_2 = Stage_x()
        self.stage_3 = Stage_x()
        self.stage_4 = Stage_x()
        self.stage_5 = Stage_x()
        self.stage_6 = Stage_x()
        for m in self.modules():
            if isinstance(m, Conv2d):
                init.constant_(m.bias, 0)
        if path:
            self.base.vgg_base.load_state_dict(torch.load(path))
        
    def forward(self, x):
        heatmaps = []
        pafs = []
        feature_map = self.base(x)
        h1, h2 = self.stage_1(feature_map)
        pafs.append(h1)
        heatmaps.append(h2)
        h1, h2 = self.stage_2(torch.cat([h1, h2, feature_map], dim = 1))
        pafs.append(h1)
        heatmaps.append(h2)
        h1, h2 = self.stage_3(torch.cat([h1, h2, feature_map], dim = 1))
        pafs.append(h1)
        heatmaps.append(h2)
        h1, h2 = self.stage_4(torch.cat([h1, h2, feature_map], dim = 1))
        pafs.append(h1)
        heatmaps.append(h2)
        h1, h2 = self.stage_5(torch.cat([h1, h2, feature_map], dim = 1))
        pafs.append(h1)
        heatmaps.append(h2)
        h1, h2 = self.stage_6(torch.cat([h1, h2, feature_map], dim = 1))
        pafs.append(h1)
        heatmaps.append(h2)
        return pafs, heatmaps        

class VGG_Base(Module):
    def __init__(self):
        super(VGG_Base, self).__init__()
        self.conv1_1 = Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv1_2 = Conv2d(in_channels = 64, out_channels = 64,  kernel_size = 3, stride = 1, padding = 1)
        self.conv2_1 = Conv2d(in_channels = 64, out_channels = 128,  kernel_size = 3, stride = 1, padding = 1)
        self.conv2_2 = Conv2d(in_channels = 128, out_channels = 128,  kernel_size = 3, stride = 1, padding = 1)
        self.conv3_1 = Conv2d(in_channels = 128, out_channels = 256,  kernel_size = 3, stride = 1, padding = 1)
        self.conv3_2 = Conv2d(in_channels = 256, out_channels = 256,  kernel_size = 3, stride = 1, padding = 1)
        self.conv3_3 = Conv2d(in_channels = 256, out_channels = 256,  kernel_size = 3, stride = 1, padding = 1)
        self.conv3_4 = Conv2d(in_channels = 256, out_channels = 256,  kernel_size = 3, stride = 1, padding = 1)
        self.conv4_1 = Conv2d(in_channels = 256, out_channels = 512,  kernel_size = 3, stride = 1, padding = 1)
        self.conv4_2 = Conv2d(in_channels = 512, out_channels = 512,  kernel_size = 3, stride = 1, padding = 1)
        self.relu = ReLU()
        self.max_pooling_2d = MaxPool2d(kernel_size = 2, stride = 2)
    
    def forward(self, x):
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.max_pooling_2d(x)
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.max_pooling_2d(x)
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        x = self.relu(self.conv3_4(x))
        x = self.max_pooling_2d(x)
        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        return x

class Base_model(Module):
    def __init__(self):
        super(Base_model, self).__init__()
        self.vgg_base = VGG_Base()
        self.conv4_3_CPM = Conv2d(in_channels=512, out_channels=256,  kernel_size = 3, stride = 1, padding = 1)
        self.conv4_4_CPM = Conv2d(in_channels=256, out_channels=128,  kernel_size = 3, stride = 1, padding = 1)
        self.relu = ReLU()
    def forward(self, x):
        x = self.vgg_base(x)
        x = self.relu(self.conv4_3_CPM(x))
        x = self.relu(self.conv4_4_CPM(x))
        return x
    
class Stage_1(Module):
    def __init__(self):
        super(Stage_1, self).__init__()
        self.conv1_CPM_L1 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2_CPM_L1 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3_CPM_L1 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4_CPM_L1 = Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv5_CPM_L1 = Conv2d(in_channels=512, out_channels=38, kernel_size=1, stride=1, padding=0)
        self.conv1_CPM_L2 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2_CPM_L2 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3_CPM_L2 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4_CPM_L2 = Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv5_CPM_L2 = Conv2d(in_channels=512, out_channels=19, kernel_size=1, stride=1, padding=0)
        self.relu = ReLU()
        
    def forward(self, x):
        h1 = self.relu(self.conv1_CPM_L1(x)) # branch1
        h1 = self.relu(self.conv2_CPM_L1(h1))
        h1 = self.relu(self.conv3_CPM_L1(h1))
        h1 = self.relu(self.conv4_CPM_L1(h1))
        h1 = self.conv5_CPM_L1(h1)
        h2 = self.relu(self.conv1_CPM_L2(x)) # branch2
        h2 = self.relu(self.conv2_CPM_L2(h2))
        h2 = self.relu(self.conv3_CPM_L2(h2))
        h2 = self.relu(self.conv4_CPM_L2(h2))
        h2 = self.conv5_CPM_L2(h2)
        return h1, h2
    
class Stage_x(Module):
    def __init__(self):
        super(Stage_x, self).__init__()
        self.conv1_L1 = Conv2d(in_channels = 185, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.conv2_L1 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.conv3_L1 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.conv4_L1 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.conv5_L1 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.conv6_L1 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 1, stride = 1, padding = 0)
        self.conv7_L1 = Conv2d(in_channels = 128, out_channels = 38, kernel_size = 1, stride = 1, padding = 0)
        self.conv1_L2 = Conv2d(in_channels = 185, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.conv2_L2 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.conv3_L2 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.conv4_L2 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.conv5_L2 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.conv6_L2 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 1, stride = 1, padding = 0)
        self.conv7_L2 = Conv2d(in_channels = 128, out_channels = 19, kernel_size = 1, stride = 1, padding = 0)
        self.relu = ReLU()
        
    def forward(self, x):
        h1 = self.relu(self.conv1_L1(x)) # branch1
        h1 = self.relu(self.conv2_L1(h1))
        h1 = self.relu(self.conv3_L1(h1))
        h1 = self.relu(self.conv4_L1(h1))
        h1 = self.relu(self.conv5_L1(h1))
        h1 = self.relu(self.conv6_L1(h1))
        h1 = self.conv7_L1(h1)
        h2 = self.relu(self.conv1_L2(x)) # branch2
        h2 = self.relu(self.conv2_L2(h2))
        h2 = self.relu(self.conv3_L2(h2))
        h2 = self.relu(self.conv4_L2(h2))
        h2 = self.relu(self.conv5_L2(h2))
        h2 = self.relu(self.conv6_L2(h2))
        h2 = self.conv7_L2(h2)
        return h1, h2

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F

from .pointnet_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction

class FeatureExtraction(nn.Module):
    def __init__(self, normal_channel=False, mask_channel=False, out_dim=256):
        super(FeatureExtraction, self).__init__()
        in_channel = 0
        if mask_channel:
            in_channel += 1
        if normal_channel:
            in_channel += 3

        self.ext_channel = in_channel

        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320, [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)

    def forward(self, xyz):
        B, C, N = xyz.shape
        if self.ext_channel > 0:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        x = l3_points.view(B, -1)

        x = F.relu(self.bn1(self.fc1(x)))
        point_feat = F.relu(self.bn2(self.fc2(x)))
        return point_feat, l3_points

class UniNet(nn.Module):
    def __init__(self, num_class=2, feat_dim=512, mask_channel=False, normal_channel=False, only_test=False):
        super(UniNet, self).__init__()

        self.feat_ext = FeatureExtraction(normal_channel=normal_channel,
                                          mask_channel=mask_channel,
                                          out_dim=feat_dim)

        self.fc1 = nn.Linear(feat_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(0.3) # drop 30% during training
        self.cls = nn.Linear(128, num_class)
        # loss function
        self.CELoss = nn.CrossEntropyLoss()
        self.only_test = only_test

    def forward(self, sample):
        points = sample['data']

        feat, _ = self.feat_ext(points)
        feat1 = F.relu(self.bn1(self.fc1(feat)))
        feat2 = F.relu(self.bn2(self.fc2(feat1)))
        feat2 = self.drop(feat2)
        pred = self.cls(feat2)
        prob = torch.softmax(pred, dim=1)
        if self.only_test:
            return {'prob': prob}

        gt = sample['label']
        loss = self.get_loss(pred, gt)

        return {'loss': loss, 'prob': prob}

    def get_loss(self, pred_logits, gt_labels):
        loss = self.CELoss(pred_logits, gt_labels)
        return loss


if __name__ == '__main__':
    # setting 1
    point_w_mask = torch.ones((2, 4, 513))
    model1 = UniNet(mask_channel=True)
    pred = model1(point_w_mask)
    print(pred.shape)

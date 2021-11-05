import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import pytorch3d.transforms as torch_transform

from .pointnet_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction

def assert_not_nan(x, info):
    assert not torch.isnan(x).sum(), info

class FeatureExtraction(nn.Module):
    def __init__(self, normal_channel=False, mask_channel=False, out_dim=128):
        super(FeatureExtraction, self).__init__()
        in_channel = 0
        if mask_channel:
            in_channel += 1
        if normal_channel:
            in_channel += 3

        self.ext_channel = in_channel

        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel, [[16, 16, 32], [32, 32, 64], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 160, [[32, 32, 64], [64, 64, 128], [64, 64, 128]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 320 + 3, [128, 256, 512], True)

        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)

    def forward(self, xyz):
        B, C, N = xyz.shape
        if self.ext_channel > 0:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        l1_xyz, l1_points = self.sa1(xyz, norm)

        assert_not_nan(l1_xyz, '!l1_xyz, xyz: {} {}'.format(
            xyz.min(), xyz.max()))
        assert_not_nan(l1_points, '!l1_points, xyz: {} {}'.format(
            xyz.min(), xyz.max()
        ))

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)

        assert_not_nan(l2_xyz, '!l2_xyz, l1_xyz: {} {}, l1_point: {} {}'.format(
            l1_xyz.min(), l1_xyz.max(), l1_points.min(), l1_points.max()))

        assert_not_nan(l2_points, '!l2_point, l1_xyz: {} {}, l1_point: {} {}'.format(
            l1_xyz.min(), l1_xyz.max(), l1_points.min(), l1_points.max()))

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        assert_not_nan(l3_xyz, '!l3_xyz, l2_xyz: {} {}, l2_point: {} {}'.format(
            l2_xyz.min(), l2_xyz.max(), l2_points.min(), l2_points.max()))

        assert_not_nan(l3_points, '!l3_points, l2_xyz: {} {}, l2_point: {} {}'.format(
            l2_xyz.min(), l2_xyz.max(), l2_points.min(), l2_points.max()))

        x = l3_points.view(B, -1)

        x = F.relu(self.bn1(self.fc1(x)))

        assert_not_nan(x, '!pre_l, l3: {} {}'.format(
            l3_points.min(), l3_points.max()
        ))

        point_feat = F.relu(self.bn2(self.fc2(x)))

        assert_not_nan(point_feat, '!point feat, x: {} {}'.format(
            x.min(), x.max()
        ))

        return point_feat, l3_points

class VNet(nn.Module):
    def __init__(self, sup_feat=128, obj_feat=128, z_feat=64, z_dim=3, rot_rep='6d',
                 mask_channel=False, normal_channel=False, only_test=False):
        super(VNet, self).__init__()

        self.rotation_rep = rot_rep

        self.sup_feat_ext = FeatureExtraction(normal_channel=normal_channel,
                                              mask_channel=mask_channel,
                                              out_dim=sup_feat)
        self.obj_feat_ext = FeatureExtraction(normal_channel=normal_channel,
                                              mask_channel=mask_channel,
                                              out_dim=obj_feat)

        self.z_feat_ext = nn.Sequential(*[nn.Conv1d(z_dim, 32, kernel_size=(1, )),
                                          nn.BatchNorm1d(32),
                                          nn.ReLU(),
                                          nn.Conv1d(32, z_feat, kernel_size=(1, )),
                                          nn.BatchNorm1d(z_feat),
                                          nn.ReLU()])


        # feature fusing
        self.fc1 = nn.Conv1d(sup_feat+obj_feat+64, 128, kernel_size=(1, ))
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Conv1d(128, 64, kernel_size=(1, ))
        self.bn2 = nn.BatchNorm1d(64)

        self.drop = nn.Dropout(0.1) # drop 10% during training

        if rot_rep == 'axis_angle':
            d = 6
        elif rot_rep == '6d':
            d = 9
        else:
            raise NotImplementedError

        self.pose_reg = nn.Sequential(*[nn.Conv1d(64, 32, kernel_size=(1,)),
                                        nn.Conv1d(32, d, kernel_size=(1,))])

        self.only_test = only_test

    def forward(self, samp_dict):
        '''

        :param sup_points: (B, 3, N)
        :param obj_points: (B, 3, N)
        :param z: (B, C, M)
        :return:
        '''

        sup_points = samp_dict['support']
        obj_points = samp_dict['object']
        z = samp_dict['z_noise']

        assert not torch.isnan(sup_points).sum(), '# {}, {} contain NaN'.format(samp_dict['sup_path'], sup_points.shape)
        assert not torch.isnan(obj_points).sum(), '# {}, {} contain NaN'.format(samp_dict['obj_path'], obj_points.shape)

        sup_feat, _ = self.sup_feat_ext(sup_points) # (B, C1)

        assert not torch.isnan(sup_feat).sum(), '# Support {} [{} {}] feature contain NaN.'.format(sup_points.shape,
                                                                                                   sup_points.min(),
                                                                                                   sup_points.max()
                                                                                                   )

        obj_feat, _ = self.obj_feat_ext(obj_points) # (B, C2)

        assert not torch.isnan(obj_feat).sum(), '# Object {} [{} {}] feature contain NaN.'.format(obj_points.shape,
                                                                                                  obj_points.min(),
                                                                                                  obj_points.max()
                                                                                                  )

        z_feat = self.z_feat_ext(z) # (B, C3, M)

        assert not torch.isnan(z_feat).sum(), '# Z feature contain NaN.'

        M = z_feat.shape[2]

        sup_feat_rpt = sup_feat.unsqueeze(2).repeat((1, 1, M))
        obj_feat_rpt = obj_feat.unsqueeze(2).repeat((1, 1, M))

        fuse_feat = torch.cat([sup_feat_rpt, obj_feat_rpt, z_feat], dim=1)
        feat1 = F.relu(self.bn1(self.fc1(fuse_feat)))

        assert not torch.isnan(feat1).sum(), '# Deep Feature1 contain NaN.'

        feat2 = F.relu(self.bn2(self.fc2(feat1)))

        assert not torch.isnan(feat2).sum(), '# Deep Feature2 contain NaN.'

        pred = self.pose_reg(feat2) # (B, 6, M)

        assert not torch.isnan(pred).sum(), '# Raw predictions contain NaN.'

        pred_transforms = self.compute_transforms(pred, self.rotation_rep)

        assert not torch.isnan(pred_transforms).sum(), '# Transforms contain NaN.'

        ret = {'pred': pred_transforms}
        if self.only_test:
            return ret

        p2g_loss, g2p_loss, pred_pc, selt_pc = self.get_projection_loss(pred_transforms=pred_transforms,
                                                                        gt_transforms=samp_dict['transforms'],
                                                                        object_pc=obj_points)

        loss = p2g_loss + g2p_loss
        ret['loss'] = loss
        ret['pred_pc'] = pred_pc
        ret['selt_pc'] = selt_pc

        assert not torch.isnan(loss).sum(), '# Loss contain NaN.'
        return ret

    def compute_transforms(self, pred, rep=''):
        '''
        :param pred: (B, n, M)
        :return: (B, M, 4, 4)
        '''
        B, d, M = pred.shape
        pred = pred.permute(0, 2, 1) # (B, M, 6)
        pred_trs = pred[..., :3].unsqueeze(3) # (B, M, 3, 1)

        if rep == 'axis_angle':
            assert d == 6, pred.shape
            pred_rot = torch_transform.axis_angle_to_matrix(pred[..., 3:])
        elif rep == '6d':
            assert d == 9, pred.shape
            pred_rot = torch_transform.rotation_6d_to_matrix(pred[..., 3:])
        else:
            raise NotImplementedError

        transform = torch.cat([pred_rot, pred_trs], dim=3) # (B, M, 3, 4)
        ones = torch.tensor([0, 0, 0, 1],
                            device=transform.device).view(1, 1, 1, 4)
        ones = ones.repeat((B, M, 1, 1))
        transform = torch.cat([transform, ones], dim=2) # (B, M, 4, 4)
        return transform

    def get_projection_loss(self, pred_transforms: torch.Tensor, gt_transforms: torch.Tensor, object_pc: torch.Tensor):
        '''

        :param pred_transforms: (B, M1, 4, 4)
        :param gt_transforms: (B, M2, 4, 4)
        :param object_pc: (B, 3, N)
        :return:
        '''


        N = object_pc.shape[2]
        B = object_pc.shape[0]
        M1 = pred_transforms.shape[1]
        M2 = gt_transforms.shape[1]

        ones = torch.ones((B, 1, N), device=object_pc.device)
        object_pc = torch.cat([object_pc, ones], dim=1).permute(0, 2, 1).unsqueeze(1).unsqueeze(4) # (B, 1, N, 4, 1)

        gt_object_ = torch.matmul(gt_transforms.unsqueeze(2), object_pc)[..., :3, 0] # (B, M2, N, 3)
        pred_object_ = torch.matmul(pred_transforms.unsqueeze(2), object_pc)[..., :3, 0] # (B, M1, N, 3)

        pred_object = pred_object_.unsqueeze(2) # (B, M1, 1, N, 3)
        gt_object = gt_object_.unsqueeze(1) # (B, 1, M2, N, 3)

        m1m2_dist = torch.sum((pred_object - gt_object) ** 2, dim=4, keepdim=False) # (B, M1, M2, N)

        m1m2_dist = torch.mean(m1m2_dist, dim=3, keepdim=False) # (B, M1, M2)

        p2g_loss, g_ind = torch.min(m1m2_dist, dim=2, keepdim=False) # (B, M1)
        g2p_loss, p_ind = torch.min(m1m2_dist, dim=1, keepdim=False) # (B, M2)

        # arrange gt object using selected index
        g_ind = g_ind.view((B, M1, 1, 1)).repeat((1, 1, N, 3))
        selt_object = torch.gather(gt_object_, dim=1, index=g_ind) # (B, M1, N, 3)
        #
        return torch.mean(p2g_loss), torch.mean(g2p_loss), pred_object_, selt_object

    def get_direct_loss(self, pred_transforms: torch.Tensor, gt_transforms: torch.Tensor, object_pc: torch.Tensor):
        '''

        :param pred_transforms: (B, M1, 4, 4)
        :param gt_transforms: (B, M2, 4, 4)
        :param object_pc: (B, 3, N)
        :return:
        '''
        N = object_pc.shape[2]
        B = object_pc.shape[0]
        pred_transforms = pred_transforms.unsqueeze(2)
        gt_transforms = gt_transforms.unsqueeze(1)
        m1m2_dist = torch.sum(torch.sum((pred_transforms - gt_transforms) ** 2,
                                        dim=4, keepdim=False), dim=3, keepdim=False) # (B, M1, M2)

        p2g_loss, _ = torch.min(m1m2_dist, dim=2, keepdim=False)
        g2p_loss, _ = torch.min(m1m2_dist, dim=1, keepdim=False)
        return torch.mean(p2g_loss), torch.mean(g2p_loss)


if __name__ == '__main__':
    import torch.optim as optim

    point1 = torch.ones((2, 3, 320))  #
    point2 = torch.ones((2, 3, 200))  #
    z_noise = torch.ones((2, 4, 7))
    # setting 2
    model = VNet(mask_channel=False, rot_rep='6d',
                 z_dim=4, obj_feat=16, sup_feat=16)

    gt_trans = torch.eye(4).unsqueeze(0).unsqueeze(1).repeat((2, 3, 1, 1)) # (2, 3, 4, 4)
    gt_trans[0, 0, 2, 3] = 1
    print('gt shape: ', gt_trans.shape)
    print(gt_trans[0, 0])
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    sample = {'support': point1, 'object': point2,
              'z_noise': z_noise, 'transforms': gt_trans}

    for ep in range(1000):
        optimizer.zero_grad()
        ret = model.forward(sample)
        tot_loss = ret['loss']
        T = ret['pred']
        print('step: ', ep)
        print('loss: ', tot_loss)
        print('T: ', T[0, 0])
        print('pred_pc: ', ret['pred_pc'].shape)
        print('selt_pc: ', ret['selt_pc'].shape)
        tot_loss.backward()
        optimizer.step()


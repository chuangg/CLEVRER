import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F


### Propagation Networks

class RelationEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RelationEncoder, self).__init__()

        self.conv1 = nn.Conv2d(input_size, hidden_size, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_size*2, hidden_size*3, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(hidden_size*3, output_size, kernel_size=3, stride=1, padding=0)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        '''
        args:
            x: [n_relations, input_size]
        returns:
            [n_relations, output_size]
        '''
        # 24 x 24
        x = self.relu(self.conv1(x))
        # 12 x 12
        x = self.relu(self.conv2(x))
        # 6 x 6
        x = self.relu(self.conv3(x))
        # 3 x 3
        x = self.relu(self.conv4(x))

        return x.view(x.size(0), -1)


class ParticleEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ParticleEncoder, self).__init__()

        self.conv1 = nn.Conv2d(input_size, hidden_size, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_size*2, hidden_size*3, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(hidden_size*3, output_size, kernel_size=3, stride=1, padding=0)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        '''
        args:
            x: [n_particles, input_size]
        returns:
            [n_particles, output_size]
        '''
        # 24 x 24
        x_1 = self.relu(self.conv1(x))
        # 12 x 12
        x_2 = self.relu(self.conv2(x_1))
        # 6 x 6
        x_3 = self.relu(self.conv3(x_2))
        # 3 x 3
        x_4 = self.relu(self.conv4(x_3))

        return x_1, x_2, x_3, x_4


class Propagator(nn.Module):
    def __init__(self, input_size, output_size, residual=False):
        super(Propagator, self).__init__()

        self.residual = residual

        self.linear = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, res=None):
        '''
        Args:
            x: [n_relations/n_particles, input_size]
        Returns:
            [n_relations/n_particles, output_size]
        '''
        if self.residual:
            x = self.relu(self.linear(x) + res)
        else:
            x = self.relu(self.linear(x))

        return x


class ParticlePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ParticlePredictor, self).__init__()

        self.convt1 = nn.ConvTranspose2d(input_size*2, hidden_size*3, kernel_size=3, stride=1, padding=0)
        self.convt2 = nn.ConvTranspose2d(hidden_size*3*2, hidden_size*2, kernel_size=4, stride=2, padding=1)
        self.convt3 = nn.ConvTranspose2d(hidden_size*2*2, hidden_size*1, kernel_size=4, stride=2, padding=1)
        self.convt4 = nn.ConvTranspose2d(hidden_size*1*2, output_size, kernel_size=4, stride=2, padding=1)
        self.relu = nn.LeakyReLU()

    def forward(self, x, x_encode):
        '''
        Args:
            x: [n_particles, input_size]
        Returns:
            [n_particles, output_size]
        '''
        x = self.relu(self.convt1(torch.cat([x, x_encode[3]], 1)))
        x = self.relu(self.convt2(torch.cat([x, x_encode[2]], 1)))
        x = self.relu(self.convt3(torch.cat([x, x_encode[1]], 1)))
        x = self.convt4(torch.cat([x, x_encode[0]], 1))

        return x


class RelationPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RelationPredictor, self).__init__()

        self.linear_0 = nn.Linear(input_size, hidden_size)
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        '''
        Args:
            x: [n_particles, input_size]
        Returns:
            [n_particles, output_size]
        '''
        x = self.relu(self.linear_0(x))
        x = self.relu(self.linear_1(x))
        x = self.linear_2(x)

        return x


class PropagationNetwork(nn.Module):
    def __init__(self, args, residual=False, use_gpu=False):

        super(PropagationNetwork, self).__init__()

        self.args = args

        input_dim = args.state_dim * (args.n_his + 1)
        relation_dim = args.relation_dim * (args.n_his + 1)
        output_dim = args.state_dim

        nf_particle = args.nf_particle
        nf_relation = args.nf_relation
        nf_effect = args.nf_effect

        self.nf_effect = args.nf_effect

        self.use_attr = args.use_attr

        self.use_gpu = use_gpu
        self.residual = residual

        # (1) state
        if args.use_attr:
            self.particle_encoder = ParticleEncoder(
                input_dim + args.attr_dim, nf_particle, nf_effect)
        else:
            self.particle_encoder = ParticleEncoder(
                input_dim, nf_particle, nf_effect)

        # (1) state receiver (2) state_sender
        if args.use_attr:
            self.relation_encoder = RelationEncoder(
                2 * input_dim + 2 * args.attr_dim + relation_dim, nf_relation, nf_effect)
        else:
            self.relation_encoder = RelationEncoder(
                2 * input_dim + relation_dim, nf_relation, nf_effect)

        # (1) relation encode (2) sender effect (3) receiver effect
        self.relation_propagator = Propagator(3 * nf_effect, nf_effect)

        # (1) particle encode (2) particle effect
        self.particle_propagator = Propagator(2 * nf_effect, nf_effect, self.residual)

        # (1) particle effect
        self.particle_predictor = ParticlePredictor(nf_effect, nf_particle, output_dim)

        # (1) relation effect
        self.relation_predictor = RelationPredictor(nf_effect, nf_effect, 1)

    def forward(self, attr, state, Rr, Rs, Ra, node_r_idx, node_s_idx, pstep, ret_feat=False):

        # print("attr size", attr.size())
        # print("state size", state.size())

        # calculate particle encoding
        if self.use_gpu:
            particle_effect = Variable(torch.zeros((state.size(0), self.nf_effect)).cuda())
        else:
            particle_effect = Variable(torch.zeros((state.size(0), self.nf_effect)))

        Rrp = Rr.t()
        Rsp = Rs.t()

        n_relation_r, n_object_r = Rrp.size(0), Rrp.size(1)
        n_relation_s, n_object_s = Rsp.size(0), Rsp.size(1)
        n_attr, n_state, bbox_h, bbox_w = attr.size(1), state.size(1), state.size(2), state.size(3)

        # receiver_state, sender_state
        attr_r = attr[node_r_idx]
        attr_s = attr[node_s_idx]
        attr_r_rel = torch.mm(Rrp, attr_r.view(n_object_r, -1)).view(n_relation_r, n_attr, bbox_h, bbox_w)
        attr_s_rel = torch.mm(Rsp, attr_s.view(n_object_s, -1)).view(n_relation_s, n_attr, bbox_h, bbox_w)
        state_r = state[node_r_idx]
        state_s = state[node_s_idx]
        state_r_rel = torch.mm(Rrp, state_r.view(n_object_r, -1)).view(n_relation_r, n_state, bbox_h, bbox_w)
        state_s_rel = torch.mm(Rsp, state_s.view(n_object_s, -1)).view(n_relation_s, n_state, bbox_h, bbox_w)

        # particle encode
        if self.use_attr:
            particle_encode = self.particle_encoder(torch.cat([attr_r, state_r], 1))
        else:
            particle_encode = self.particle_encoder(state_r)
        # print("particle encode:",
        # particle_encode[0].size(), particle_encode[1].size(),
        # particle_encode[2].size(), particle_encode[3].size())

        # calculate relation encoding
        if self.use_attr:
            relation_encode = self.relation_encoder(
                torch.cat([attr_r_rel, attr_s_rel, state_r_rel, state_s_rel, Ra], 1))
        else:
            relation_encode = self.relation_encoder(
                torch.cat([state_r_rel, state_s_rel, Ra], 1))
        # print("relation encode:", relation_encode.size())

        for i in range(pstep):
            # print("pstep", i)

            # print("Receiver index range", np.min(node_r_idx), np.max(node_r_idx))
            # print("Sender index range", np.min(node_s_idx), np.max(node_s_idx))

            effect_p_r = particle_effect[node_r_idx]
            effect_p_s = particle_effect[node_s_idx]

            receiver_effect = Rrp.mm(effect_p_r)
            sender_effect = Rsp.mm(effect_p_s)

            # calculate relation effect
            # print(relation_encode.size())
            # print(receiver_effect.size())
            # print(sender_effect.size())
            effect_rel = self.relation_propagator(
                torch.cat([relation_encode, receiver_effect, sender_effect], 1))
            # print("relation effect:", effect_rel.size())

            # calculate particle effect by aggregating relation effect
            effect_p_r_agg = Rr.mm(effect_rel)

            # calculate particle effect
            effect_p = self.particle_propagator(
                torch.cat([particle_encode[-1].view(particle_encode[-1].size(0), -1), effect_p_r_agg], 1),
                res=effect_p_r)
            # print("particle effect:", effect_p.size())

            particle_effect[node_r_idx] = effect_p

        ### predict for object
        pred_obj = self.particle_predictor(
            particle_effect.view(particle_effect.size(0), particle_effect.size(1), 1, 1),
            particle_encode)

        # average position channel
        pred_obj[:, 1] = torch.mean(pred_obj[:, 1].view(pred_obj.size(0), -1), 1).view(pred_obj.size(0), 1, 1)
        pred_obj[:, 2] = torch.mean(pred_obj[:, 2].view(pred_obj.size(0), -1), 1).view(pred_obj.size(0), 1, 1)
        # print("pred_obj:", pred_obj.size())

        ### predict for relation
        pred_rel = self.relation_predictor(effect_rel)
        # print("pred_rel:", pred_rel.size())

        if ret_feat:
            return pred_obj, pred_rel, particle_effect
        else:
            return pred_obj, pred_rel


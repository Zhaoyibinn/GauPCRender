import torch
import torch.nn as nn
import torch.nn.functional as F

class MlpConv(nn.Module):
    def __init__(self, input_channel, channels, activation_function=None):
        super(MlpConv, self).__init__()
        self.layer_num = len(channels)
        self.net = nn.Sequential()
        last_channel = input_channel
        for i, channel in enumerate(channels):   
            self.net.add_module('Conv1d_%d' % i, nn.Conv1d(last_channel, channel, kernel_size=1))
            if i != self.layer_num - 1:
                self.net.add_module('ReLU_%d' % i, nn.ReLU())
            last_channel = channel
        if activation_function != None:
            self.net.add_module('af', activation_function)

    def forward(self, x):
        return self.net(x)        


class PcnEncoder(nn.Module):
    def __init__(self, input_channel=3, out_c=1024):
        super().__init__()
        self.mlp_conv_1 = MlpConv(input_channel, [128, 256])
        self.mlp_conv_2 = MlpConv(512, [512, out_c])

    def forward(self, x):
        '''
        x : [B, N, 3]
        '''
        B, N, _ = x.shape
        x = x.permute(0, 2, 1)
        x = self.mlp_conv_1(x)

        x_max = torch.max(x, 2, keepdim=True).values
        x_max = x_max.repeat(1, 1, N) 
        x = torch.cat([x, x_max], 1)
        
        x = self.mlp_conv_2(x)
        
        x_max = torch.max(x, 2, keepdim=True).values
        return x_max
    

class TopNetNode(nn.Module):
    def __init__(self, input_channel, append_channel, output_channel, output_num, activation_function=None):
        super(TopNetNode, self).__init__()
        self.append_channel = append_channel
        self.output_channel = output_channel 
        self.output_num = output_num
        self.mlp_conv = MlpConv(input_channel+append_channel, [512, 256, 64, output_channel*output_num], activation_function=activation_function)
    
    '''
    append_x shape: [bs, feature_channel, 1]
    '''
    def forward(self, x, append_x=None):
        batch_size = x.shape[0]
        point_num = x.shape[2]
        if self.append_channel != 0:
            append_x = append_x.repeat(1, 1, point_num)
            x = torch.cat([x, append_x], 1)
        x = self.mlp_conv(x)
        x = torch.reshape(x, (batch_size, self.output_channel, -1))
        return x


class TopNetDecoder(nn.Module):
    def __init__(self, input_channel, output_nums, get_all_res=False):
        super(TopNetDecoder, self).__init__()
        self.get_all_res = get_all_res
        self.topnet_node_0 = TopNetNode(input_channel, 0, 8, output_nums[0], activation_function=nn.Tanh())
        self.topnet_nodes = []
        
        for output_num in output_nums[1:-1]:
            self.topnet_nodes.append(TopNetNode(8, input_channel, 8, output_num))
        self.topnet_nodes.append(TopNetNode(8, input_channel, 3, output_nums[-1]))
        self.topnet_nodes = nn.ModuleList(self.topnet_nodes)

    '''
    x shape: [bs, feature_channel, 1] or [bs, feature_channel]
    '''
    def forward(self, x):
        node_res = []
        if len(x.shape) == 2:
            global_x = torch.unsqueeze(x, 2)
        else:
            assert(len(x.shape) == 3 and x.shape[2] == 1)
            global_x = x
        res = self.topnet_node_0(global_x)
        node_res.append(res)
        for topnet_node in self.topnet_nodes:
            res = topnet_node(res, global_x)
            node_res.append(res)
        res = torch.permute(res, [0, 2, 1])
        if self.get_all_res:
            return res, node_res
        else:
            return res
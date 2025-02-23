# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Temporal modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import autopad
__all__ = "ConvLSTMCell", "YOLOConvLSTM"

# class Conv(nn.Module):
#     """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

#     default_act = nn.SiLU()  # default activation

#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
#         """Initialize Conv layer with given arguments including activation."""
#         super().__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

#     def forward(self, x):
#         """Apply convolution, batch normalization and activation to input tensor."""
#         return self.act(self.bn(self.conv(x)))

#     def forward_fuse(self, x):
#         """Apply convolution and activation without batch normalization."""
#         return self.act(self.conv(x))

class ConvLSTMCell(nn.Module):
    # def __init__(self, input_dim, hidden_dim, kernel_size=1, stride=1, padding=None, groups=1, dilation=1, activation_function=True):

        
        # self.input_dim = input_dim
        # self.hidden_dim = hidden_dim
        # self.kernel_size = kernel_size
        # self.stride = stride
        # self.padding = padding
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):

        super().__init__()
        
        self.input_dim = c1
        self.hidden_dim = c2
        self.kernel_size = k
        self.stride = s
        self.groups = g
        self.dilation = d
        self.padding = p

        # Combined gates for efficiency
        # self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        # self.conv = nn.Conv2d(
        #     in_channels=c1+c2,
        #     out_channels=4 * c2,  # 4 gates: input, forget, cell, output
        #     kernel_size=k,
        #     padding=autopad(k, p, d),
        #     groups=g, 
        #     dilation=d,
        #     bias=True
        # )
        self.bn = nn.BatchNorm2d(c2)
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,  # 4 gates: input, forget, cell, output
            kernel_size=self.kernel_size,
            padding=self.padding,
            groups=self.groups, 
            dilation=self.dilation,
            bias=True
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        # Concatenate input and hidden state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        # Calculate gates
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        # Apply gates
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

class YOLOConvLSTM(nn.Module):
    def __init__(self, c1, c2, k=1, p=1, num_layers=1):
        super(YOLOConvLSTM, self).__init__()
        
        self.input_channels = c1
        self.hidden_channels = c2
        self.kernel_size = k
        self.num_layers = num_layers
        self.padding = p
        
        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_channels if i == 0 else self.hidden_channels
            cell_list.append(
                ConvLSTMCell(
                    c1=cur_input_dim,
                    c2=self.hidden_channels,
                    k=self.kernel_size,
                    p=self.padding
                )
            )
        self.cell_list = nn.ModuleList(cell_list)
        
        # Additional conv layer to match original channel dimensions 
        self.conv_out = nn.Conv2d(c2, c1, 1)
        
    def forward(self, x, hidden_state=None):
        batch_size, _, height, width = x.size()
        
        # Initialize hidden states
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size, height, width)
        
        layer_output = x
        new_hidden_states = []
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            h, c = self.cell_list[layer_idx](layer_output, (h, c))
            layer_output = h
            new_hidden_states.append((h, c))
        
        # Match original channel dimensions
        output = self.conv_out(layer_output)
        
        return output, new_hidden_states
    
    def _init_hidden(self, batch_size, height, width):
        hidden_states = []
        for _ in range(self.num_layers):
            h = torch.zeros(batch_size, self.hidden_channels, height, width).to(next(self.parameters()).device)
            c = torch.zeros(batch_size, self.hidden_channels, height, width).to(next(self.parameters()).device)
            hidden_states.append((h, c))
        return hidden_states
    
# class YOLOTemporalNeck(nn.Module):
#     def __init__(self, neck_channels):
#         super().__init__()
#         # ConvLSTM for each feature level
#         self.convlstm_p3 = YOLOConvLSTM(input_channels=neck_channels[0], hidden_channels=neck_channels[0]//2)
#         self.convlstm_p4 = YOLOConvLSTM(input_channels=neck_channels[1], hidden_channels=neck_channels[1]//2)
#         self.convlstm_p5 = YOLOConvLSTM(input_channels=neck_channels[2], hidden_channels=neck_channels[2]//2)
        
#         # Hidden state storage
#         self.hidden_states = {
#             'p3': None,
#             'p4': None,
#             'p5': None
#         }

#     def forward(self, x, reset_states=False):
#         if reset_states:
#             self.hidden_states = {k: None for k in self.hidden_states}

#         # Process each feature level with ConvLSTM
#         temporal_features = []
#         for i, (level_feat, convlstm) in enumerate(zip(x, [self.convlstm_p3, self.convlstm_p4, self.convlstm_p5])):
#             temporal_feat, new_hidden_state = convlstm(level_feat, self.hidden_states[f'p{i+3}'])
#             self.hidden_states[f'p{i+3}'] = new_hidden_state
#             temporal_features.append(temporal_feat)

#         return temporal_features
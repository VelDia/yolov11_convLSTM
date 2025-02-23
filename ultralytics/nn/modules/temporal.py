# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Temporal modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import autopad
__all__ = "ConvLSTMCell", "YOLOConvLSTM"
from ultralytics.utils.torch_utils import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    intersect_dicts,
    model_info,
    scale_img,
    time_sync,
)
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.plotting import feature_visualization

# # class Conv(nn.Module):
# #     """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

# #     default_act = nn.SiLU()  # default activation

# #     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
# #         """Initialize Conv layer with given arguments including activation."""
# #         super().__init__()
# #         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
# #         self.bn = nn.BatchNorm2d(c2)
# #         self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

# #     def forward(self, x):
# #         """Apply convolution, batch normalization and activation to input tensor."""
# #         return self.act(self.bn(self.conv(x)))

# #     def forward_fuse(self, x):
# #         """Apply convolution and activation without batch normalization."""
# #         return self.act(self.conv(x))

# class ConvLSTMCell(nn.Module):
#     # def __init__(self, input_dim, hidden_dim, kernel_size=1, stride=1, padding=None, groups=1, dilation=1, activation_function=True):

        
#         # self.input_dim = input_dim
#         # self.hidden_dim = hidden_dim
#         # self.kernel_size = kernel_size
#         # self.stride = stride
#         # self.padding = padding
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):

#         super().__init__()
        
#         self.input_dim = c1
#         self.hidden_dim = c2
#         self.kernel_size = k
#         self.stride = s
#         self.groups = g
#         self.dilation = d
#         self.padding = p

#         # Combined gates for efficiency
#         # self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
#         # self.conv = nn.Conv2d(
#         #     in_channels=c1+c2,
#         #     out_channels=4 * c2,  # 4 gates: input, forget, cell, output
#         #     kernel_size=k,
#         #     padding=autopad(k, p, d),
#         #     groups=g, 
#         #     dilation=d,
#         #     bias=True
#         # )
#         self.bn = nn.BatchNorm2d(c2)
#         self.conv = nn.Conv2d(
#             in_channels=self.input_dim + self.hidden_dim,
#             out_channels=4 * self.hidden_dim,  # 4 gates: input, forget, cell, output
#             kernel_size=self.kernel_size,
#             # padding=self.padding,
#             padding=(self.stride*(self.input_dim-1)+self.dilation*(self.kernel_size-1)-1)//2,
#             groups=self.groups, 
#             dilation=self.dilation,
#             bias=True
#         )

#     def forward(self, input_tensor, cur_state):
#         h_cur, c_cur = cur_state
        
#         # Concatenate input and hidden state
#         combined = torch.cat([input_tensor, h_cur], dim=1)
        
#         # Calculate gates
#         combined_conv = self.conv(combined)
#         cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
#         # Apply gates
#         i = torch.sigmoid(cc_i)
#         f = torch.sigmoid(cc_f)
#         o = torch.sigmoid(cc_o)
#         g = torch.tanh(cc_g)
        
#         c_next = f * c_cur + i * g
#         h_next = o * torch.tanh(c_next)
        
#         return h_next, c_next

# class YOLOConvLSTM(nn.Module):
#     def __init__(self, c1, c2, k=1, p=1, num_layers=1):
#         super(YOLOConvLSTM, self).__init__()
        
#         self.input_channels = c1
#         self.hidden_channels = c2
#         self.kernel_size = k
#         self.num_layers = num_layers
#         self.padding = p
        
#         cell_list = []
#         for i in range(self.num_layers):
#             cur_input_dim = self.input_channels if i == 0 else self.hidden_channels
#             cell_list.append(
#                 ConvLSTMCell(
#                     c1=cur_input_dim,
#                     c2=self.hidden_channels,
#                     k=self.kernel_size,
#                     p=self.padding
#                 )
#             )
#         self.cell_list = nn.ModuleList(cell_list)
        
#         # Additional conv layer to match original channel dimensions 
#         self.conv_out = nn.Conv2d(c2, c1, 1)
        
#     def forward(self, x, hidden_state=None):
#         batch_size, _, height, width = x.size()
        
#         # Initialize hidden states
#         if hidden_state is None:
#             hidden_state = self._init_hidden(batch_size, height, width)
        
#         layer_output = x
#         new_hidden_states = []
        
#         for layer_idx in range(self.num_layers):
#             h, c = hidden_state[layer_idx]
#             h, c = self.cell_list[layer_idx](layer_output, (h, c))
#             layer_output = h
#             new_hidden_states.append((h, c))
        
#         # Match original channel dimensions
#         output = self.conv_out(layer_output)
        
#         return output, new_hidden_states
    
#     def _init_hidden(self, batch_size, height, width):
#         hidden_states = []
#         for _ in range(self.num_layers):
#             h = torch.zeros(batch_size, self.hidden_channels, height, width).to(next(self.parameters()).device)
#             c = torch.zeros(batch_size, self.hidden_channels, height, width).to(next(self.parameters()).device)
#             hidden_states.append((h, c))
#         return hidden_states
    
# # class YOLOTemporalNeck(nn.Module):
# #     def __init__(self, neck_channels):
# #         super().__init__()
# #         # ConvLSTM for each feature level
# #         self.convlstm_p3 = YOLOConvLSTM(input_channels=neck_channels[0], hidden_channels=neck_channels[0]//2)
# #         self.convlstm_p4 = YOLOConvLSTM(input_channels=neck_channels[1], hidden_channels=neck_channels[1]//2)
# #         self.convlstm_p5 = YOLOConvLSTM(input_channels=neck_channels[2], hidden_channels=neck_channels[2]//2)
        
# #         # Hidden state storage
# #         self.hidden_states = {
# #             'p3': None,
# #             'p4': None,
# #             'p5': None
# #         }

# #     def forward(self, x, reset_states=False):
# #         if reset_states:
# #             self.hidden_states = {k: None for k in self.hidden_states}

# #         # Process each feature level with ConvLSTM
# #         temporal_features = []
# #         for i, (level_feat, convlstm) in enumerate(zip(x, [self.convlstm_p3, self.convlstm_p4, self.convlstm_p5])):
# #             temporal_feat, new_hidden_state = convlstm(level_feat, self.hidden_states[f'p{i+3}'])
# #             self.hidden_states[f'p{i+3}'] = new_hidden_state
# #             temporal_features.append(temporal_feat)

# #         return temporal_features

import torch
from torch import nn

# def initialize_weights(layer):
#         """Initialize a layer's weights and biases.

#         Args:
#             layer: A PyTorch Module's layer."""
#         if isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d)):
#             pass
#         else:
#             try:
#                 nn.init.xavier_normal_(layer.weight)
#             except AttributeError:
#                 pass
#             try:
#                 nn.init.uniform_(layer.bias)
#             except (ValueError, AttributeError):
#                 pass

class HadamardProduct(nn.Module):
    """A Hadamard product layer.
    
    Args:
        shape: The shape of the layer."""
       
    def __init__(self, shape):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(*shape))
        self.bias = nn.Parameter(torch.empty(*shape))
           
    def forward(self, x):
        return x * self.weights


    
class ConvLSTMCell(nn.Module):
    """A convolutional LSTM cell.

    Implementation details follow closely the following paper:

    Shi et al. -'Convolutional LSTM Network: A Machine Learning 
    Approach for Precipitation Nowcasting' (2015).
    Accessible at https://arxiv.org/abs/1506.04214

    The parameter names are drawn from the paper's Eq. 3.

    Args:
        input_bands: The number of bands in the input data.
        input_dim: The length of of side of input data. Data is
            presumed to have identical width and heigth."""

    def __init__(self, input_bands, input_dim, kernels=1,  dropout=0, batch_norm=True):
        super().__init__()

        self.input_bands = input_bands
        self.input_dim = input_dim
        self.kernels = kernels
        self.dropout = dropout
        self.batch_norm = batch_norm

        self.kernel_size = 3
        self.padding = 1  # Preserve dimensions

        self.input_conv_params = {
            'in_channels': self.input_bands,
            'out_channels': self.kernels,
            'kernel_size': self.kernel_size,
            'padding': self.padding,
            'bias': True
        }

        self.hidden_conv_params = {
            'in_channels': self.kernels,
            'out_channels': self.kernels,
            'kernel_size': self.kernel_size,
            'padding': self.padding,
            'bias': True
        }

        self.state_shape = (
            1,
            self.kernels,
            self.input_dim,
            self.input_dim
        )

        self.batch_norm_layer = None
        if self.batch_norm:
            self.batch_norm_layer = nn.BatchNorm2d(num_features=self.input_bands)

        # Input Gates
        self.W_xi = nn.Conv2d(**self.input_conv_params)
        self.W_hi = nn.Conv2d(**self.hidden_conv_params)
        self.W_ci = HadamardProduct(self.state_shape)

        # Forget Gates
        self.W_xf = nn.Conv2d(**self.input_conv_params)
        self.W_hf = nn.Conv2d(**self.hidden_conv_params)
        self.W_cf = HadamardProduct(self.state_shape)

        # Memory Gates
        self.W_xc = nn.Conv2d(**self.input_conv_params)
        self.W_hc = nn.Conv2d(**self.hidden_conv_params)

        # Output Gates
        self.W_xo = nn.Conv2d(**self.input_conv_params)
        self.W_ho = nn.Conv2d(**self.hidden_conv_params)
        self.W_co = HadamardProduct(self.state_shape)

        # Dropouts
        self.H_drop = nn.Dropout2d(p=self.dropout)
        self.C_drop = nn.Dropout2d(p=self.dropout)

        self.apply(initialize_weights)    

class YOLOConvLSTM(nn.Module):

    def __init__(self, input_bands, input_dim, kernels=1, dropout=0, num_layers=1, bidirectional=False):
        super().__init__()
        self.input_bands = input_bands
        self.input_dim = input_dim
        self.kernels = kernels
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        
        self.layers_fwd = self.initialize_layers()
        self.layers_bwd = None
        if self.bidirectional:
            self.layers_bwd = self.initialize_layers()
        self.fc_output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=self.kernels*self.input_dim**2*(1 if not self.bidirectional else 2), 
                out_features=1024
            ),
            nn.Linear(
                in_features=1024, 
                out_features=1
            )
        )
            
        self.apply(initialize_weights)
        
    def initialize_layers(self):
        """Initialize a single direction of the model's layers.
        
        This function takes care of stacking layers, allocating
        dropout and assigning correct input feature number for
        each layer in the stack."""
        
        layers = nn.ModuleList()
        
        for i in range(self.num_layers):
            layers.append(
                ConvLSTMCell(
                    input_bands=self.input_bands if i == 0 else self.kernels, 
                    input_dim=self.input_dim,
                    dropout=self.dropout if i+1 < self.num_layers else 0,
                    kernels=self.kernels,
                    batch_norm=False
                )
            )
            
        return layers
    
        
    def forward(self, x):
        """Perform forward pass with the model.
        
        For each item in the sequence, the data is propagated 
        through each layer and both directions, if possible.
        In case of a bidirectional model, the outputs are 
        concatenated from both directions. The output of the 
        last item of the sequence is further given to the FC
        layers to produce the final batch of predictions. 
        
        Args:
            x:  A batch of spatial data sequences. The data
                should be in the following format:
                [Batch, Seq, Band, Dim, Dim]
                    
        Returns:
            A batch of predictions."""
        
        seq_len = x.shape[1]
        
        for seq_idx in range(seq_len):
            
            layer_in_out = x[:,seq_idx,::]
            states = None
            for layer in self.layers_fwd:
                layer_in_out, states = layer(layer_in_out, states)
                
            if not self.bidirectional:
                continue
                
            layer_in_out_bwd = x[:,-seq_idx,::]
            states = None
            for layer in self.layers_bwd:
                layer_in_out_bwd, states = layer(layer_in_out_bwd, states)
            
            layer_in_out = torch.cat((layer_in_out,layer_in_out_bwd),dim=1)
            
        return self.fc_output(layer_in_out)
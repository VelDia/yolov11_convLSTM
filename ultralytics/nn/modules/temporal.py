# # # # Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# # # """Temporal modules."""

import torch
import torch.nn as nn
# # # import torch.nn.functional as F
# # # from .conv import autopad
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

# # # # class ConvLSTMCell(nn.Module):
# # # #     # def __init__(self, input_dim, hidden_dim, kernel_size=1, stride=1, padding=None, groups=1, dilation=1, activation_function=True):

        
# # # #         # self.input_dim = input_dim
# # # #         # self.hidden_dim = hidden_dim
# # # #         # self.kernel_size = kernel_size
# # # #         # self.stride = stride
# # # #         # self.padding = padding
# # # #     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):

# # # #         super().__init__()
        
# # # #         self.input_dim = c1
# # # #         self.hidden_dim = c2
# # # #         self.kernel_size = k
# # # #         self.stride = s
# # # #         self.groups = g
# # # #         self.dilation = d
# # # #         self.padding = p

# # # #         # Combined gates for efficiency
# # # #         # self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
# # # #         # self.conv = nn.Conv2d(
# # # #         #     in_channels=c1+c2,
# # # #         #     out_channels=4 * c2,  # 4 gates: input, forget, cell, output
# # # #         #     kernel_size=k,
# # # #         #     padding=autopad(k, p, d),
# # # #         #     groups=g, 
# # # #         #     dilation=d,
# # # #         #     bias=True
# # # #         # )
# # # #         self.bn = nn.BatchNorm2d(c2)
# # # #         self.conv = nn.Conv2d(
# # # #             in_channels=self.input_dim + self.hidden_dim,
# # # #             out_channels=4 * self.hidden_dim,  # 4 gates: input, forget, cell, output
# # # #             kernel_size=self.kernel_size,
# # # #             # padding=self.padding,
# # # #             padding=(self.stride*(self.input_dim-1)+self.dilation*(self.kernel_size-1)-1)//2,
# # # #             groups=self.groups, 
# # # #             dilation=self.dilation,
# # # #             bias=True
# # # #         )



# # # # class YOLOConvLSTM(nn.Module):
# # # #     def __init__(self, c1, c2, k=1, p=1, num_layers=1):
# # # #         super(YOLOConvLSTM, self).__init__()
        
# # # #         self.input_channels = c1
# # # #         self.hidden_channels = c2
# # # #         self.kernel_size = k
# # # #         self.num_layers = num_layers
# # # #         self.padding = p
        
# # # #         cell_list = []
# # # #         for i in range(self.num_layers):
# # # #             cur_input_dim = self.input_channels if i == 0 else self.hidden_channels
# # # #             cell_list.append(
# # # #                 ConvLSTMCell(
# # # #                     c1=cur_input_dim,
# # # #                     c2=self.hidden_channels,
# # # #                     k=self.kernel_size,
# # # #                     p=self.padding
# # # #                 )
# # # #             )
# # # #         self.cell_list = nn.ModuleList(cell_list)
        
# # # #         # Additional conv layer to match original channel dimensions 
# # # #         self.conv_out = nn.Conv2d(c2, c1, 1)
        
# # # #     def forward(self, x, hidden_state=None):
# # # #         batch_size, _, height, width = x.size()
        
# # # #         # Initialize hidden states
# # # #         if hidden_state is None:
# # # #             hidden_state = self._init_hidden(batch_size, height, width)
        
# # # #         layer_output = x
# # # #         new_hidden_states = []
        
# # # #         for layer_idx in range(self.num_layers):
# # # #             h, c = hidden_state[layer_idx]
# # # #             h, c = self.cell_list[layer_idx](layer_output, (h, c))
# # # #             layer_output = h
# # # #             new_hidden_states.append((h, c))
        
# # # #         # Match original channel dimensions
# # # #         output = self.conv_out(layer_output)
        
# # # #         return output, new_hidden_states
    
# # # #     def _init_hidden(self, batch_size, height, width):
# # # #         hidden_states = []
# # # #         for _ in range(self.num_layers):
# # # #             h = torch.zeros(batch_size, self.hidden_channels, height, width).to(next(self.parameters()).device)
# # # #             c = torch.zeros(batch_size, self.hidden_channels, height, width).to(next(self.parameters()).device)
# # # #             hidden_states.append((h, c))
# # # #         return hidden_states
    
# # # # # class YOLOTemporalNeck(nn.Module):
# # # # #     def __init__(self, neck_channels):
# # # # #         super().__init__()
# # # # #         # ConvLSTM for each feature level
# # # # #         self.convlstm_p3 = YOLOConvLSTM(input_channels=neck_channels[0], hidden_channels=neck_channels[0]//2)
# # # # #         self.convlstm_p4 = YOLOConvLSTM(input_channels=neck_channels[1], hidden_channels=neck_channels[1]//2)
# # # # #         self.convlstm_p5 = YOLOConvLSTM(input_channels=neck_channels[2], hidden_channels=neck_channels[2]//2)
        
# # # # #         # Hidden state storage
# # # # #         self.hidden_states = {
# # # # #             'p3': None,
# # # # #             'p4': None,
# # # # #             'p5': None
# # # # #         }

# # # # #     def forward(self, x, reset_states=False):
# # # # #         if reset_states:
# # # # #             self.hidden_states = {k: None for k in self.hidden_states}

# # # # #         # Process each feature level with ConvLSTM
# # # # #         temporal_features = []
# # # # #         for i, (level_feat, convlstm) in enumerate(zip(x, [self.convlstm_p3, self.convlstm_p4, self.convlstm_p5])):
# # # # #             temporal_feat, new_hidden_state = convlstm(level_feat, self.hidden_states[f'p{i+3}'])
# # # # #             self.hidden_states[f'p{i+3}'] = new_hidden_state
# # # # #             temporal_features.append(temporal_feat)

# # # # #         return temporal_features

# # import torch
# # from torch import nn

# # # # def initialize_weights(layer):
# # # #         """Initialize a layer's weights and biases.

# # # #         Args:
# # # #             layer: A PyTorch Module's layer."""
# # # #         if isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d)):
# # # #             pass
# # # #         else:
# # # #             try:
# # # #                 nn.init.xavier_normal_(layer.weight)
# # # #             except AttributeError:
# # # #                 pass
# # # #             try:
# # # #                 nn.init.uniform_(layer.bias)
# # # #             except (ValueError, AttributeError):
# # # #                 pass

# class HadamardProduct(nn.Module):
#     """A Hadamard product layer.
    
#     Args:
#         shape: The shape of the layer."""
       
#     def __init__(self, shape):
#         super().__init__()
#         self.weights = nn.Parameter(torch.empty(*shape))
#         self.bias = nn.Parameter(torch.empty(*shape))
           
#     def forward(self, x):
#         return x * self.weights


    
# class ConvLSTMCell(nn.Module):
#     """A convolutional LSTM cell.

#     Implementation details follow closely the following paper:

#     Shi et al. -'Convolutional LSTM Network: A Machine Learning 
#     Approach for Precipitation Nowcasting' (2015).
#     Accessible at https://arxiv.org/abs/1506.04214

#     The parameter names are drawn from the paper's Eq. 3.

#     Args:
#         input_bands: The number of bands in the input data.
#         input_dim: The length of of side of input data. Data is
#             presumed to have identical width and heigth."""

#     def __init__(self, input_bands, input_dim, kernels=1,  dropout=0, batch_norm=True):
#         super().__init__()

#         self.input_bands = input_bands
#         self.input_dim = input_dim
#         self.kernels = kernels
#         self.dropout = dropout
#         self.batch_norm = batch_norm

#         self.kernel_size = 3
#         self.padding = 1  # Preserve dimensions

#         self.input_conv_params = {
#             'in_channels': self.input_bands,
#             'out_channels': self.kernels,
#             'kernel_size': self.kernel_size,
#             'padding': self.padding,
#             'bias': True
#         }

#         self.hidden_conv_params = {
#             'in_channels': self.kernels,
#             'out_channels': self.kernels,
#             'kernel_size': self.kernel_size,
#             'padding': self.padding,
#             'bias': True
#         }

#         self.state_shape = (
#             1,
#             self.kernels,
#             self.input_dim,
#             self.input_dim
#         )

#         self.batch_norm_layer = None
#         if self.batch_norm:
#             self.batch_norm_layer = nn.BatchNorm2d(num_features=self.input_bands)

#         # Input Gates
#         self.W_xi = nn.Conv2d(**self.input_conv_params)
#         self.W_hi = nn.Conv2d(**self.hidden_conv_params)
#         self.W_ci = HadamardProduct(self.state_shape)

#         # Forget Gates
#         self.W_xf = nn.Conv2d(**self.input_conv_params)
#         self.W_hf = nn.Conv2d(**self.hidden_conv_params)
#         self.W_cf = HadamardProduct(self.state_shape)

#         # Memory Gates
#         self.W_xc = nn.Conv2d(**self.input_conv_params)
#         self.W_hc = nn.Conv2d(**self.hidden_conv_params)

#         # Output Gates
#         self.W_xo = nn.Conv2d(**self.input_conv_params)
#         self.W_ho = nn.Conv2d(**self.hidden_conv_params)
#         self.W_co = HadamardProduct(self.state_shape)

#         # Dropouts
#         self.H_drop = nn.Dropout2d(p=self.dropout)
#         self.C_drop = nn.Dropout2d(p=self.dropout)

#         self.apply(initialize_weights)    

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

#     def __init__(self, input_bands, input_dim, kernels=1, dropout=0, num_layers=1, bidirectional=False):
#         super().__init__()
#         self.input_bands = input_bands
#         self.input_dim = input_dim
#         self.kernels = kernels
#         self.num_layers = num_layers
#         self.bidirectional = bidirectional
#         self.dropout = dropout
        
#         self.layers_fwd = self.initialize_layers()
#         self.layers_bwd = None
#         if self.bidirectional:
#             self.layers_bwd = self.initialize_layers()
#         self.fc_output = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(
#                 in_features=self.kernels*self.input_dim**2*(1 if not self.bidirectional else 2), 
#                 out_features=1024
#             ),
#             nn.Linear(
#                 in_features=1024, 
#                 out_features=1
#             )
#         )
            
#         self.apply(initialize_weights)
        
#     def initialize_layers(self):
#         """Initialize a single direction of the model's layers.
        
#         This function takes care of stacking layers, allocating
#         dropout and assigning correct input feature number for
#         each layer in the stack."""
        
#         layers = nn.ModuleList()
        
#         for i in range(self.num_layers):
#             layers.append(
#                 ConvLSTMCell(
#                     input_bands=self.input_bands if i == 0 else self.kernels, 
#                     input_dim=self.input_dim,
#                     dropout=self.dropout if i+1 < self.num_layers else 0,
#                     kernels=self.kernels,
#                     batch_norm=False
#                 )
#             )
            
#         return layers
    
        
#     def forward(self, x):
#         """Perform forward pass with the model.
        
#         For each item in the sequence, the data is propagated 
#         through each layer and both directions, if possible.
#         In case of a bidirectional model, the outputs are 
#         concatenated from both directions. The output of the 
#         last item of the sequence is further given to the FC
#         layers to produce the final batch of predictions. 
        
#         Args:
#             x:  A batch of spatial data sequences. The data
#                 should be in the following format:
#                 [Batch, Seq, Band, Dim, Dim]
                    
#         Returns:
#             A batch of predictions."""
        
#         seq_len = x.shape[1]
        
#         for seq_idx in range(seq_len):
            
#             layer_in_out = x[:,seq_idx,::]
#             states = None
#             for layer in self.layers_fwd:
#                 layer_in_out, states = layer(layer_in_out, states)
                
#             if not self.bidirectional:
#                 continue
                
#             layer_in_out_bwd = x[:,-seq_idx,::]
#             states = None
#             for layer in self.layers_bwd:
#                 layer_in_out_bwd, states = layer(layer_in_out_bwd, states)
            
#             layer_in_out = torch.cat((layer_in_out,layer_in_out_bwd),dim=1)
            
#         return self.fc_output(layer_in_out)

# class ConvLSTMCell(nn.Module):

#     def __init__(self, input_dim, hidden_dim, kernel_size=(1, 1), bias=True):
#         """
#         Initialize ConvLSTM cell.
#         Parameters
#         ----------
#         input_dim: int
#             Number of channels of input tensor.
#         hidden_dim: int
#             Number of channels of hidden state.
#         kernel_size: (int, int)
#             Size of the convolutional kernel.
#         bias: bool
#             Whether or not to add the bias.
#         """

#         super(ConvLSTMCell, self).__init__()

#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim

#         self.kernel_size = kernel_size
#         self.padding = kernel_size[0] // 2, kernel_size[1] // 2
#         self.bias = bias

#         self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
#                               out_channels=4 * self.hidden_dim,
#                               kernel_size=self.kernel_size,
#                               padding=self.padding,
#                               bias=self.bias)

#     def forward(self, input_tensor, cur_state):
#         h_cur, c_cur = cur_state

#         combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

#         combined_conv = self.conv(combined)
#         cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
#         i = torch.sigmoid(cc_i)
#         f = torch.sigmoid(cc_f)
#         o = torch.sigmoid(cc_o)
#         g = torch.tanh(cc_g)

#         c_next = f * c_cur + i * g
#         h_next = o * torch.tanh(c_next)

#         return h_next, c_next

#     def init_hidden(self, batch_size, image_size):
#         height, width = image_size
#         return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
#                 torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


# class YOLOConvLSTM(nn.Module):

#     """
#     Parameters:
#         input_dim: Number of channels in input
#         hidden_dim: Number of hidden channels
#         kernel_size: Size of kernel in convolutions
#         num_layers: Number of LSTM layers stacked on each other
#         batch_first: Whether or not dimension 0 is the batch or not
#         bias: Bias or no bias in Convolution
#         return_all_layers: Return the list of computations for all layers
#         Note: Will do same padding.
#     Input:
#         A tensor of size B, T, C, H, W or T, B, C, H, W
#     Output:
#         A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
#             0 - layer_output_list is the list of lists of length T of each output
#             1 - last_state_list is the list of last states
#                     each element of the list is a tuple (h, c) for hidden state and memory
#     Example:
#         >> x = torch.rand((32, 10, 64, 128, 128))
#         >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
#         >> _, last_states = convlstm(x)
#         >> h = last_states[0][0]  # 0 for layer index, 0 for h index
#     """

#     def __init__(self, input_dim, hidden_dim, kernel_size=(1,1), num_layers=1,
#                  batch_first=False, bias=True, return_all_layers=False):
#         super(YOLOConvLSTM, self).__init__()

#         # self._check_kernel_size_consistency(kernel_size)

#         # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
#         kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
#         hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
#         if not len(kernel_size) == len(hidden_dim) == num_layers:
#             raise ValueError('Inconsistent list length.')

#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.kernel_size = kernel_size
#         self.num_layers = num_layers
#         self.batch_first = batch_first
#         self.bias = bias
#         self.return_all_layers = return_all_layers

#         cell_list = []
#         for i in range(0, self.num_layers):
#             cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

#             cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
#                                           hidden_dim=self.hidden_dim[i],
#                                           kernel_size=self.kernel_size[i],
#                                           bias=self.bias))

#         self.cell_list = nn.ModuleList(cell_list)


#     def forward(self, input_tensor, hidden_state=None):
#         """
#         Parameters
#         ----------
#         input_tensor: todo
#             5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
#         hidden_state: todo
#             None. todo implement stateful
#         Returns
#         -------
#         last_state_list, layer_output
#         """
#         if not self.batch_first:
#             # (t, b, c, h, w) -> (b, t, c, h, w)
#             input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

#         b, _, _, h, w = input_tensor.size()

#         # Implement stateful ConvLSTM
#         if hidden_state is not None:
#             raise NotImplementedError()
#         else:
#             # Since the init is done in forward. Can send image size here
#             hidden_state = self._init_hidden(batch_size=b,
#                                              image_size=(h, w))

#         layer_output_list = []
#         last_state_list = []

#         seq_len = input_tensor.size(1)
#         cur_layer_input = input_tensor

#         for layer_idx in range(self.num_layers):

#             h, c = hidden_state[layer_idx]
#             output_inner = []
#             for t in range(seq_len):
#                 h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
#                                                  cur_state=[h, c])
#                 output_inner.append(h)

#             layer_output = torch.stack(output_inner, dim=1)
#             cur_layer_input = layer_output

#             layer_output_list.append(layer_output)
#             last_state_list.append([h, c])

#         if not self.return_all_layers:
#             layer_output_list = layer_output_list[-1:]
#             last_state_list = last_state_list[-1:]

#         return layer_output_list, last_state_list

#     def _init_hidden(self, batch_size, image_size):
#         init_states = []
#         for i in range(self.num_layers):
#             init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
#         return init_states

#     @staticmethod
#     def _check_kernel_size_consistency(kernel_size):
#         if not (isinstance(kernel_size, tuple) or
#                 (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
#             raise ValueError('`kernel_size` must be tuple or list of tuples')

#     @staticmethod
#     def _extend_for_multilayer(param, num_layers):
#         if not isinstance(param, list):
#             param = [param] * num_layers
#         return param

# import torch
# import torch.nn as nn

# Original ConvLSTM cell as proposed by Shi et al.
class ConvLSTMCell(nn.Module):
    default_act = torch.relu
    def __init__(self, in_channels, out_channels, 
    kernel_size=1, padding=1, activation=True, frame_size=(640, 640)):
        
        super(ConvLSTMCell, self).__init__()  
        self.act = self.default_act if activation is True else activation if isinstance(activation, nn.Module) else nn.Identity()
        # if activation == "tanh":
        #     self.activation = torch.tanh 
        # elif activation == "relu":
        #     self.activation = torch.relu
        
        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        self.conv = nn.Conv2d(
            in_channels=in_channels + out_channels, 
            out_channels=4 * out_channels, 
            kernel_size=kernel_size, 
            padding=padding)           

        # Initialize weights for Hadamard Products
        self.W_ci = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_co = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_cf = nn.Parameter(torch.Tensor(out_channels, *frame_size))

    def forward(self, X, H_prev, C_prev):

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        conv_output = self.conv(torch.cat([X, H_prev], dim=1))

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)

        input_gate = torch.sigmoid(i_conv + self.W_ci * C_prev )
        forget_gate = torch.sigmoid(f_conv + self.W_cf * C_prev )

        # Current Cell output
        C = forget_gate*C_prev + input_gate * self.activation(C_conv)

        output_gate = torch.sigmoid(o_conv + self.W_co * C )

        # Current Hidden State
        H = output_gate * self.activation(C)

        return H, C
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class YOLOConvLSTM(nn.Module):

    def __init__(self, in_channels, out_channels, 
    kernel_size=1, padding=1, activation = True, frame_size=(640, 640)):

        super(YOLOConvLSTM, self).__init__()

        self.out_channels = out_channels

        # We will unroll this over time steps
        self.convLSTMcell = ConvLSTMCell(in_channels, out_channels, 
        kernel_size, padding, activation, frame_size)

    def forward(self, X):

        # X is a frame sequence (batch_size, num_channels, seq_len, height, width)

        # Get the dimensions
        batch_size, _, seq_len, height, width = X.size()

        # Initialize output
        output = torch.zeros(batch_size, self.out_channels, seq_len, 
        height, width, device=device)
        
        # Initialize Hidden State
        H = torch.zeros(batch_size, self.out_channels, 
        height, width, device=device)

        # Initialize Cell Input
        C = torch.zeros(batch_size,self.out_channels, 
        height, width, device=device)

        # Unroll over time steps
        for time_step in range(seq_len):

            H, C = self.convLSTMcell(X[:,:,time_step], H, C)

            output[:,:,time_step] = H

        return output

# import torch
# from torch import nn
# import torch.nn.functional as f
# from torch.autograd import Variable


# # Define some constants
# KERNEL_SIZE = 3
# PADDING = KERNEL_SIZE // 2


# class ConvLSTMCell(nn.Module):
#     """
#     Generate a convolutional LSTM cell
#     """

#     def __init__(self, input_size, hidden_size):
#         super().__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, KERNEL_SIZE, padding=PADDING)

#     def forward(self, input_, prev_state):

#         # get batch and spatial sizes
#         batch_size = input_.data.size()[0]
#         spatial_size = input_.data.size()[2:]

#         # generate empty prev_state, if None is provided
#         if prev_state is None:
#             state_size = [batch_size, self.hidden_size] + list(spatial_size)
#             prev_state = (
#                 Variable(torch.zeros(state_size)),
#                 Variable(torch.zeros(state_size))
#             )

#         prev_hidden, prev_cell = prev_state

#         # data size is [batch, channel, height, width]
#         stacked_inputs = torch.cat((input_, prev_hidden), 1)
#         gates = self.Gates(stacked_inputs)

#         # chunk across channel dimension
#         in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

#         # apply sigmoid non linearity
#         in_gate = f.sigmoid(in_gate)
#         remember_gate = f.sigmoid(remember_gate)
#         out_gate = f.sigmoid(out_gate)

#         # apply tanh non linearity
#         cell_gate = f.tanh(cell_gate)

#         # compute current cell and hidden state
#         cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
#         hidden = out_gate * f.tanh(cell)

#         return hidden, cell


# def _main():
#     """
#     Run some basic tests on the API
#     """

#     # define batch_size, channels, height, width
#     b, c, h, w = 1, 3, 4, 8
#     d = 5           # hidden state size
#     lr = 1e-1       # learning rate
#     T = 6           # sequence length
#     max_epoch = 20  # number of epochs

#     # set manual seed
#     torch.manual_seed(0)

#     print('Instantiate model')
#     model = ConvLSTMCell(c, d)
#     print(repr(model))

#     print('Create input and target Variables')
#     x = Variable(torch.rand(T, b, c, h, w))
#     y = Variable(torch.randn(T, b, d, h, w))

#     print('Create a MSE criterion')
#     loss_fn = nn.MSELoss()

#     print('Run for', max_epoch, 'iterations')
#     for epoch in range(0, max_epoch):
#         state = None
#         loss = 0
#         for t in range(0, T):
#             state = model(x[t], state)
#             loss += loss_fn(state[0], y[t])

#         print(' > Epoch {:2d} loss: {:.3f}'.format((epoch+1), loss.data[0]))

#         # zero grad parameters
#         model.zero_grad()

#         # compute new grad parameters through time!
#         loss.backward()

#         # learning_rate step against the gradient
#         for p in model.parameters():
#             p.data.sub_(p.grad.data * lr)

#     print('Input size:', list(x.data.size()))
#     print('Target size:', list(y.data.size()))
#     print('Last hidden state size:', list(state[0].size()))


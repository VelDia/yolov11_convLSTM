import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, padding):
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = padding
        
        # Combined gates for efficiency
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,  # 4 gates: input, forget, cell, output
            kernel_size=self.kernel_size,
            padding=self.padding,
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
    def __init__(self, input_channels, hidden_channels, kernel_size=3, num_layers=1):
        super(YOLOConvLSTM, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.padding = kernel_size // 2
        
        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_channels if i == 0 else self.hidden_channels
            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_channels,
                    kernel_size=self.kernel_size,
                    padding=self.padding
                )
            )
        self.cell_list = nn.ModuleList(cell_list)
        
        # Additional conv layer to match original channel dimensions if needed
        self.conv_out = nn.Conv2d(hidden_channels, input_channels, kernel_size=1)
        
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
    

# Example integration point
convlstm_p3 = YOLOConvLSTM(input_channels=256, hidden_channels=128)  # For P3 feature level
convlstm_p4 = YOLOConvLSTM(input_channels=512, hidden_channels=256)  # For P4 feature level
convlstm_p5 = YOLOConvLSTM(input_channels=1024, hidden_channels=512)  # For P5 feature level
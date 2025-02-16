"""
Hypernetwork: Training for projecting the higher dimension to lower dimension.
"""
import torch
import torch.nn as nn

class Knowledge_translator(nn.Module):
    def __init__(self, translate_modules, n_layers, input_dim, output_dim):
        """Knowledge translator network.
        
        Args:
            translate_modules: List of module names to translate
            n_layers: Number of layers in the model
            input_dim: Input dimension (teacher model)
            output_dim: Output dimension (student model)
        """
        super().__init__()
        self.model_mapping = {
            "down_proj": "mlp",
            "up_proj": "mlp", 
            "v_proj": "self_attn",
            "o_proj": "self_attn",
        }
        
        self.network = nn.ModuleDict({
            m: nn.Sequential(
                nn.Linear(input_dim, input_dim, bias=False, dtype=torch.float32),
                nn.ReLU(),
                nn.Linear(input_dim, output_dim, bias=False, dtype=torch.float32)
            )
            for m in translate_modules
        })
        self.n_layers = n_layers
        self.network.apply(self.init_weights)
        
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=1e-5)
            
    def forward(self, x, source_idxs):
        result = {}
        for k, v in x.items():
            if k in ["down_proj", "o_proj"]:
                v_reshaped = v.transpose(1, 2)
                network_input = v_reshaped.to(torch.float32)
                network_output = self.network[k](network_input)
                result[k] = network_output.to(torch.float16).transpose(1, 2)
            else:
                network_input = v.to(torch.float32)
                network_output = self.network[k](network_input)
                result[k] = network_output.to(torch.float16)
        return result

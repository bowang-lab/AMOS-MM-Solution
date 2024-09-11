from torch import nn
from .spatial_pooling_projector import SpatialPoolingProjector
from .hilt_projector import HILTProjector
from .token_packer import TokenPacker

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, *args, **kwargs):
        return x
    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class Minigpt(nn.Module):
    def __init__(self, config=None):
        super(Minigpt, self).__init__()
        # c*4 is the input size, and c is the output size for the linear layer
        inc, ouc = config.mm_hidden_size, config.hidden_size
        self.linear = nn.Linear(inc * 4, ouc)

    def forward(self, x):
        # x is the input tensor with shape [b, num_tokens, c]
        b, num_tokens, c = x.shape

        # Check if num_tokens is divisible by 4
        if num_tokens % 4 != 0:
            raise ValueError("num_tokens must be divisible by 4")

        # Reshape x to [b, num_tokens/4, c*4]
        x = x.view(b, num_tokens // 4, c * 4)

        # Apply the linear transformation
        x = self.linear(x)
        return x


class Vanilla(nn.Module):
    def __init__(self, config=None):
        super(Vanilla, self).__init__()
        # c*4 is the input size, and c is the output size for the linear layer
        inc, ouc = config.mm_hidden_size, config.hidden_size
        self.linear = nn.Linear(inc * 4, ouc)

    def forward(self, x):
        b, num_tokens, c = x.shape

        # Check if num_tokens is divisible by 4
        if num_tokens % 4 != 0:
            raise ValueError("num_tokens must be divisible by 4")

        # First, reshape to [b, num_tokens//4, 4, c]
        x = x.view(b, num_tokens // 4, 4, c)

        # Then, permute to interleave the tokens
        x = x.permute(0, 1, 3, 2).contiguous()

        # Finally, reshape to [b, num_tokens//4, c*4] to interleave features of 4 tokens
        x = x.view(b, num_tokens // 4, c * 4)

        # Apply the linear transformation
        x = self.linear(x)
        return x

class FullLinear(nn.Module):
    def __init__(self, config):
        super(FullLinear, self).__init__()
        self.linear = nn.Linear(config.mm_hidden_size, config.hidden_size)
    def forward(self, x):
        x = self.linear(x)
        return x
    @property
    def proj_out_num(self):
        num = 2048
        return num
    
class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()

        # calculate multipler for num of tokens
        self.concat_after_project = config.concat_after_project
        self.multipler = config.multipler
        self.out = 256

        # calcualte the num of tokens projecter will receive
        tokens = 1
        if isinstance(config.any_res_crops, list):
            config_ = config.any_res_crops[0]
            D = config.image_size[0] // config.patch_size[0] 
            H = config.image_size[1] // config.patch_size[1]
            W = config.image_size[2] // config.patch_size[2]
            tokens = D * H * W
        else:
            for a, b in zip(config.image_size, config.patch_size):
                tokens *= (a // b)

        if not self.concat_after_project:
            tokens *= self.multipler

        self.embedding_projection = nn.Linear(config.mm_hidden_size, config.hidden_size)
        depth = int(config.proj_layer_num)
        modules = [nn.Linear(int(tokens), self.out)]
        for _ in range(0, depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(self.out, self.out))
        self.token_projector = nn.Sequential(*modules)

    def forward(self, x):
        batch_size, num_tokens, embedding_dim = x.shape
        x = x.reshape(-1, embedding_dim)
        x = self.embedding_projection(x)
        x = x.reshape(batch_size, num_tokens, -1)

        x = x.transpose(1, 2)
        x = self.token_projector(x)
        x = x.transpose(1, 2)

        return x

    @property
    def proj_out_num(self):
        n = self.out 
        if self.concat_after_project:
            n *= self.multipler
        return n


def build_mm_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type')

    if projector_type == 'linear':
        return FullLinear(config)
    elif projector_type == 'mlp':
        return MLP(config)
    elif projector_type == 'spp':
        return SpatialPoolingProjector(config, image_size=config.image_size,
                                        patch_size=config.patch_size,
                                        in_dim=config.mm_hidden_size,
                                        out_dim=config.hidden_size,
                                        layer_type=config.proj_layer_type,
                                        layer_num=config.proj_layer_num,
                                        pooling_type=config.proj_pooling_type,
                                        pooling_size=config.proj_pooling_size)
    elif projector_type == 'tp':
        return TokenPacker(hidden_size=config.hidden_size)
    elif projector_type == 'hilt':
        return HILTProjector(
            layer_num=config.proj_layer_num,
            hidden_size=config.mm_hidden_size,
            out_dim=config.hidden_size,
            proj_out_num=config.proj_out_num

        )
    elif projector_type == 'identity':
        return IdentityMap()
    else:
        raise ValueError(f'Unknown projector type: {projector_type}')
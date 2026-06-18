import math
import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

class Image2Token(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, visual_hidden_size=1280, text_hidden_size=768, max_length=77, num_layers=3):
        super(Image2Token, self).__init__()
        
        self.visual_proj = nn.Linear(visual_hidden_size, text_hidden_size)
        self.text_hidden_size = text_hidden_size
        
        if num_layers>0:
            self.query = nn.Parameter(torch.randn((1, max_length, text_hidden_size)))
            decoder_layer = nn.TransformerDecoderLayer(d_model=text_hidden_size, nhead=text_hidden_size//64, batch_first=True)
            self.i2t = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        else:
            self.i2t = None

    def forward(self, x):
        b=x.shape[0]
        out = self.visual_proj(x).view(b,-1,self.text_hidden_size)
        if self.i2t is not None:
            out = self.i2t(self.query.repeat(b,1,1), out)

        return out
    

    
class ID2Token(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, id_dim=512, text_hidden_size=768, max_length=77, num_layers=3):
        super(ID2Token, self).__init__()
        
        self.id_proj = nn.Linear(id_dim, text_hidden_size)
        self.text_hidden_size = text_hidden_size
        
        if num_layers>0:
            self.query = nn.Parameter(torch.randn((1, max_length, text_hidden_size)))
            decoder_layer = nn.TransformerDecoderLayer(d_model=text_hidden_size, nhead=text_hidden_size//64, batch_first=True)
            self.id2t = nn.TransformerDecoder(decoder_layer, num_layers=num_layers-1)
            
            # 时间步骤编码层
            self.time_embed = nn.Sequential(
                nn.Linear(text_hidden_size, 4 * text_hidden_size),
                nn.SiLU(),
                nn.Linear(4 * text_hidden_size, text_hidden_size),
            )
            
            # 时间步骤注入的自注意力层
            encoder_layer = nn.TransformerEncoderLayer(d_model=text_hidden_size, nhead=text_hidden_size//64, batch_first=True)
            self.time_attention = nn.TransformerEncoder(encoder_layer, num_layers=num_layers-1)
            
            # 最后一层Transformer
            decoder_layer_final = nn.TransformerDecoderLayer(d_model=text_hidden_size, nhead=text_hidden_size//64, batch_first=True)
            self.final_layer = nn.TransformerDecoder(decoder_layer_final, num_layers=1)
        else:
            self.id2t = None
            self.time_attention1 = None
            self.time_attention2 = None
            self.final_layer = None

    def forward(self, x, timesteps):
        b=x.shape[0]
        out = self.id_proj(x).view(b,-1,self.text_hidden_size)
        if self.id2t is not None:
            # 前两层Transformer处理
            out = self.id2t(self.query.repeat(b,1,1), out)
            
            # 时间步骤编码
            t_emb = timesteps.float()
            t_emb = t_emb * 1000.0  # 缩放时间步骤
            t_emb = t_emb.view(-1, 1)  # [B, 1]
            half_dim = self.text_hidden_size // 2
            emb = math.log(10000.0) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, device=t_emb.device) * -emb)
            emb = t_emb * emb
            t_emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
            t_emb = self.time_embed(t_emb)
            
            # 时间步骤通过两层自注意力处理
            t_emb = t_emb.unsqueeze(1).repeat(1, out.size(1), 1)
            t_emb = self.time_attention(t_emb)
            
            # 特征相加
            out = out + t_emb
            
            # 最后一层Transformer处理
            out = self.final_layer(self.query.repeat(b,1,1), out)

        return out


def test_id2token():
    # 设置随机种子以保证可重复性
    torch.manual_seed(42)
    
    # 创建模型实例
    model = ID2Token(id_dim=512, text_hidden_size=768, max_length=77, num_layers=3)
    
    # 生成测试数据
    batch_size = 4
    input_data = torch.randn(batch_size, 64, 512)  # bs*49*512
    timesteps = torch.randint(0, 1000, (batch_size,))  # bs，用于时间步骤编码，范围0-1000
    print(timesteps)
    # 前向传播
    output = model(input_data, timesteps)
    
    # 打印输出形状
    print(f"Input shape: {input_data.shape}")
    print(f"Timesteps shape: {timesteps.shape}")
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    test_id2token()


class ID2TokenEncoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, id_dim=512, text_hidden_size=1024, num_layers=3):
        super(ID2TokenEncoder, self).__init__()
        
        self.id_proj = nn.Linear(id_dim, text_hidden_size)
        self.text_hidden_size = text_hidden_size
        
        if num_layers>0:
            encoder_layer = nn.TransformerEncoderLayer(d_model=text_hidden_size, nhead=text_hidden_size//64, batch_first=True)
            self.id2t = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            self.id2t = None

    def forward(self, x):
        b=x.shape[0]
        out = self.id_proj(x).view(b,-1,self.text_hidden_size)
        if self.id2t is not None:
            out = self.id2t(out)

        return out
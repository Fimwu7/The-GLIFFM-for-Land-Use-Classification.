import torch
import torch.nn as nn

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, norm_cfg=None, act_cfg=None,
                 bias=True, groups=1):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias,
                              groups=groups)
        self.norm = None
        if norm_cfg and norm_cfg['type'] == 'BN2d':
            self.norm = nn.BatchNorm2d(out_channels)
        self.act = None
        if act_cfg and act_cfg['type'] == 'GELU':
            self.act = nn.GELU()
        elif act_cfg and act_cfg['type'] == 'ReLU':
            self.act = nn.ReLU(inplace=True)
        elif act_cfg and act_cfg['type'] == 'Sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x

class GIF(nn.Module):
    def __init__(self, channel, fusion_groups=4):
        super(GIF, self).__init__()
        self.fusion_groups = fusion_groups
        self.channels_per_group = channel // self.fusion_groups
        self.group_gate = nn.ModuleList([
            ConvModule(self.channels_per_group, 1, kernel_size=3, padding=1, act_cfg=dict(type='Sigmoid'),
                       norm_cfg=None)
            for _ in range(self.fusion_groups)
        ])
        self.group_conv = nn.Sequential(
           nn.Conv2d(channel, channel, kernel_size=1, padding=0),
       )

    def forward(self, fused_initial):
        _, _, H_fused, W_fused = fused_initial.shape

        chunks = torch.chunk(fused_initial, self.fusion_groups,
                             dim=1)


        gated_chunks = []
        for i, chunk in enumerate(chunks):
            gate = self.group_gate[i](chunk)
            gated_chunk = chunk * gate
            gated_chunks.append(gated_chunk)


        gated_fused = torch.cat(gated_chunks, dim=1)
        x = self.group_conv(gated_fused)+gated_fused
        return x


if __name__ == '__main__':
    feature = torch.randn(32, 48, 256, 256)
    model = GIF(channel=48, fusion_groups=4)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params / 1e6:.2f} M")
    out = model(feature)
    print(f"Output shape: {out.size()}")

import torch
import torch.nn as nn


class LGFEM(nn.Module):
    def __init__(self, channel):
        super(LGFEM, self).__init__()
        self.channel = channel
        self.output_channel = 2 * channel

        self.la_conv = nn.Sequential(
            nn.Conv2d(channel // 2, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel // 2, kernel_size=3, padding=1),
            nn.GELU()
        )

        self.lb_conv = nn.Sequential(
            nn.Conv2d(channel // 2, channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel // 2, kernel_size=1, padding=0),
            nn.GELU()
        )

        self.l_weight_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.g_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channel, channel),
            nn.GELU(),
            nn.Linear(channel, channel),
            nn.GELU(),
            nn.Unflatten(1, (channel, 1, 1))
        )

        self.g_conv = nn.Sequential(
            nn.Conv2d(channel, channel // 4, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 4, channel // 2, kernel_size=1, padding=0),
        )

        self.avgpool_g = nn.AdaptiveAvgPool2d(1)
        self.maxpool_g = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, l, g):
        #  Local feature processing
        l_max = l.amax(dim=1, keepdim=True)
        l_avg = l.mean(dim=1, keepdim=True)
        l_weight = self.l_weight_conv(torch.cat([l_max, l_avg], dim=1))

        la_input, lb_input = torch.chunk(l, 2, dim=1)
        la_output = self.la_conv(la_input)
        lb_output = self.lb_conv(lb_input)
        l_processed = torch.cat([la_output, lb_output], dim=1)
        l_final = l_processed * l_weight

        # Global feature processing
        g_avg_pooled = self.avgpool_g(g)
        g_max_pooled = self.maxpool_g(g)

        g_avg_weights_raw = self.g_conv(g_avg_pooled)
        g_max_weights_raw = self.g_conv(g_max_pooled)

        g_channel_weights = self.sigmoid(torch.cat([g_avg_weights_raw, g_max_weights_raw], dim=1))

        ga = g * g_channel_weights

        gb = self.g_fc(g)
        g_final = torch.add(ga, gb)
        return torch.cat((l_final, g_final), dim=1)


if __name__ == '__main__':
    m = LGFEM(channel=24)
    a = torch.randn(1, 24, 256, 256)
    b = torch.randn(1, 24, 256, 256)
    c = m(a, b)
    print(c.size())

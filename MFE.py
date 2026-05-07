from torch import nn
import torch


class MFE(nn.Module):
    def __init__(self, channel):
        super(MFE, self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(channel // 2, channel // 4, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(channel // 4, channel // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.Conf = nn.Sequential(
            nn.Conv2d(channel, channel // 4, kernel_size=1, padding=0),
        )
        self.final = nn.Conv2d(channel, channel, kernel_size=3)

    def forward(self, rgb, lcmap):
        l1, l2 = torch.chunk(lcmap, 2, dim=1)
        r1, r2 = torch.chunk(rgb, 2, dim=1)
        c1 = torch.cat((self.Conv(l1), l2), dim=1)
        c1 = self.Conf(c1)
        c2 = torch.cat((self.Conv(l2), r1), dim=1)
        c2 = self.Conf(c2)
        c3 = torch.cat((self.Conv(r1), r2), dim=1)
        c3 = self.Conf(c3)
        c4 = torch.cat((self.Conv(r2), l1), dim=1)
        c4 = self.Conf(c4)
        x = torch.cat((c1, c2, c3, c4), dim=1)
        x = self.final(x)
        return x


if __name__ == '__main__':
    lcmap = torch.randn(32, 48, 256, 256)
    # The lcmap serves as the prediction mask produced by the segmentation network.
    rgb = torch.randn(32, 48, 256, 256)
    model = MFE(channel=48)
    out = model(lcmap, rgb)
    print(out.size())

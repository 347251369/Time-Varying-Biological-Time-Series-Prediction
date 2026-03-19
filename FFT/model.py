from torch import nn
import torch
import numpy as np
import math

class FourierFilter(nn.Module):
    def __init__(self):
        super(FourierFilter, self).__init__()

    def forward(self, x, alpha):
        xf = torch.fft.fft(x, dim=1)
        mask = torch.ones_like(xf)
        mask[:,int((1-alpha)*xf.shape[1]):,:] = 0
        x_l = torch.fft.ifft(xf*mask, dim=1).real
        x_g = x - x_l
        return x_g, x_l

class Encoder(nn.Module):
    def __init__(self, D, d):
        super(Encoder, self).__init__()
        self.D = D
        self.d = d
        self.half = int(self.D + self.d)
        self.fc1 = nn.Linear(self.D, self.half)
        self.fc2 = nn.Linear(self.half, self.half)
        self.fc3 = nn.Linear(self.half, self.d)
        self.f = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0) 

    def forward(self, x):
        x = self.f(self.fc1(x))
        x = self.f(self.fc2(x))
        x = self.fc3(x)
        return x

class Decoder(nn.Module):
    def __init__(self, D, d):
        super(Decoder, self).__init__()
        self.D = D
        self.d = d
        self.half = int(self.D+self.d)
        self.fc1 = nn.Linear(self.d, self.half)
        self.fc2 = nn.Linear(self.half, self.half)
        self.fc3 = nn.Linear(self.half, self.D)
        self.f = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0) 
    def forward(self, x):
        x = self.f(self.fc1(x))
        x = self.f(self.fc2(x))
        x = self.fc3(x)
        return x

class Global_module(nn.Module):
    def __init__(self, D, d):
        super(Global_module, self).__init__()
        self.D = D
        self.d = d
        self.Encoder_g = Encoder(self.D, self.d)
        self.Decoder_g = Decoder(self.D, self.d)
        self.Koopman_g = nn.Linear(self.d, self.d, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        K_init = torch.randn(self.d, self.d)
        U, _, V = torch.svd(K_init)
        self.Koopman_g.weight.data = torch.mm(U, V.t())

    def forward(self, X):
        x = self.Encoder_g(X)
        i_X = self.Decoder_g(x)

        y = self.Koopman_g(x)
        n_Y = self.Decoder_g(y)
        return i_X, n_Y

class TV_module(nn.Module):
    def __init__(self, D, d):
        super(TV_module, self).__init__()
        self.D = D
        self.d = d
        self.Encoder_g = Encoder(self.D, self.d)
        self.Decoder_g = Decoder(self.D, self.d)

    def forward(self, X, Y):
        x = self.Encoder_g(X)
        i_X = self.Decoder_g(x)

        y = self.Encoder_g(Y)
        i_Y = self.Decoder_g(y)

        K = torch.matmul(torch.linalg.pinv(x),y)
        z = torch.matmul(y,K).real
        n_Z = self.Decoder_g(z)

        return i_X, i_Y, n_Z

class TVNN(nn.Module):
    def __init__(self, D):
        super(TVNN, self).__init__()
        self.D = D
        self.d = int(self.D*2)
        self.fft = FourierFilter()
        self.Global_module = Global_module(self.D,self.d)
        self.TV_module = TV_module(self.D,self.d)

    def forward(self, x, y,alpha):
        x_g, x_l = self.fft(x,alpha)
        y_g, y_l = self.fft(y,alpha)

        ix_g, ny_g = self.Global_module(x_g)
        iy_g, nz_g = self.Global_module(y_g)
        ix_l, iy_l, nz_l = self.TV_module(x_l,y_l)

        return ix_g, ny_g, iy_g , nz_g , ix_l, iy_l, nz_l, nz_g + nz_l
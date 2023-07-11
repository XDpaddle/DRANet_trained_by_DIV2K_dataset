'''
import torch
import torch.nn.functional as F
from torch.autograd import Variable
'''
import paddle
import paddle.nn.functional as F
import numpy as np
from math import exp

"""
# ============================================
# SSIM loss
# https://github.com/Po-Hsun-Su/pytorch-ssim
# ============================================
"""


def gaussian(window_size, sigma):
    #gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    gauss = paddle.to_tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)],dtype='float32')
    return gauss/gauss.sum()


def create_window(window_size, channel):#window_size=11
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    #_2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    _2D_window = paddle.to_tensor(_1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0),dtype='float32')
    #window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    window = paddle.to_tensor(paddle.expand(_2D_window,shape=[channel, 1, window_size, window_size]))
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIMLoss(paddle.nn.Layer):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.shape
        if channel == self.channel and self.window.dtype == img1.dtype():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            #if img1.is_cuda:
            #    window = window.cuda(img1.get_device())
            #window = window.type_as(img1)
            window = paddle.to_tensor(window, dtype=img1.dtype)
            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.shape
    window = create_window(window_size, channel)
    
    #if img1.is_cuda:
     #   window = window.cuda(img1.get_device())
    #window = window.type_as(img1)
    window=paddle.to_tensor(window,dtype=img1.dtype)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)


if __name__ == '__main__':
    import cv2
    #from torch import optim
    from skimage import io
    npImg1 = cv2.imread("C:\\Users\\wang\\Desktop/renxiang.jpg")

    img1 = paddle.to_tensor(np.rollaxis(npImg1, 2),dtype='float32').unsqueeze(0)/255.0
    #img2 = paddle.rand(img1.size())
    img2 = paddle.rand(img1.shape)
    '''
    img1 = Variable(img1, requires_grad=False)
    img2 = Variable(img2, requires_grad=True)'''

    ssim_value = ssim(img1, img2).item()
    print("Initial ssim:", ssim_value)

    ssim_loss = SSIMLoss()
    #optimizer = paddle.optimizer.Adam([img2], learning_rate=0.01)
    optimizer = paddle.optimizer.Adam(learning_rate=0.01,parameters=[img2])
    while ssim_value < 0.99:
        optimizer.clear_grad()
        ssim_out = -ssim_loss(img1, img2)
        ssim_value = -ssim_out.item()
        print('{:<4.4f}'.format(ssim_value))
        ssim_out.backward()
        optimizer.step()
    img = np.transpose(img2.detach().cpu().squeeze().float().numpy(), (1,2,0))
    io.imshow(np.uint8(np.clip(img*255, 0, 255)))

from mxnet import gluon, nd
from mxnet.gluon import nn
import numpy as np
from datetime import datetime
import time
from scipy.ndimage import maximum_filter1d
import cv2
from matplotlib.colors import hsv_to_rgb

nfeat = 128
sz  = [3, 3, 3, 3, 3]
sz2 = [3, 3, 3, 3, 3]
szf = [1]

def plot_flows(y):
    Y = (np.clip(normalize99(y[0][0]),0,1) - 0.5) * 2
    X = (np.clip(normalize99(y[1][0]),0,1) - 0.5) * 2
    H = (np.arctan2(Y, X) + np.pi) / (2*np.pi)
    S = normalize99(y[0][0]**2 + y[1][0]**2)
    HSV = np.concatenate((H[:,:,np.newaxis], S[:,:,np.newaxis], S[:,:,np.newaxis]), axis=-1)
    HSV = np.clip(HSV, 0.0, 1.0)
    flow = (hsv_to_rgb(HSV) * 255).astype(np.uint8)
    return flow

def plot_outlines(masks):
    outpix = []
    contours, hierarchy = cv2.findContours(masks.astype(np.int32), mode=cv2.RETR_FLOODFILL, method=cv2.CHAIN_APPROX_SIMPLE)
    for c in range(len(contours)):
        pix = contours[c].astype(int).squeeze()
        if len(pix)>4:
            peri = cv2.arcLength(contours[c], True)
            approx = cv2.approxPolyDP(contours[c], 0.001, True)[:,0,:]
            outpix.append(approx)
    return outpix

def plot_overlay(img, masks):
    img = normalize99(img.astype(np.float32).mean(axis=-1))
    img -= img.min()
    img /= img.max()
    HSV = np.zeros((img.shape[0], img.shape[1], 3), np.float32)
    HSV[:,:,2] = np.clip(img*1.5, 0, 1.0)
    for n in range(int(masks.max())):
        ipix = (masks==n+1).nonzero()
        HSV[ipix[0],ipix[1],0] = np.random.rand()
        HSV[ipix[0],ipix[1],1] = 1.0
    RGB = (hsv_to_rgb(HSV) * 255).astype(np.uint8)
    return RGB

def image_resizer(img, resize=512):
    ny,nx = img.shape[:2]
    if np.array(img.shape).max() > resize:
        if ny>nx:
            nx = int(nx/ny * resize)
            ny = resize
        else:
            ny = int(ny/nx * resize)
            nx = resize
        shape = (nx,ny)
        img = cv2.resize(img, shape)
        img = img.astype(np.uint8)
    return img

def extendROI(ypix, xpix, Ly, Lx,niter=1):
    for k in range(niter):
        yx = ((ypix, ypix, ypix, ypix-1, ypix+1), (xpix, xpix+1,xpix-1,xpix,xpix))
        yx = np.array(yx)
        yx = yx.reshape((2,-1))
        yu = np.unique(yx, axis=1)
        ix = np.all((yu[0]>=0, yu[0]<Ly, yu[1]>=0 , yu[1]<Lx), axis = 0)
        ypix,xpix = yu[:, ix]
    return ypix,xpix

def get_mask(y, rpad=20, nmax=20):
    xp = y[1,:,:].flatten().astype('int32')
    yp = y[0,:,:].flatten().astype('int32')
    _, Ly, Lx = y.shape
    xm, ym = np.meshgrid(np.arange(Lx),  np.arange(Ly))

    xedges = np.arange(-.5-rpad, xm.shape[1]+.5+rpad, 1)
    yedges = np.arange(-.5-rpad, xm.shape[0]+.5+rpad, 1)
    #xp = (xm-dx).flatten().astype('int32')
    #yp = (ym-dy).flatten().astype('int32')
    h,_,_ = np.histogram2d(xp, yp, bins=[xedges, yedges])

    hmax = maximum_filter1d(h, 5, axis=0)
    hmax = maximum_filter1d(hmax, 5, axis=1)

    yo, xo = np.nonzero(np.logical_and(h-hmax>-1e-6, h>10))
    Nmax = h[yo, xo]
    isort = np.argsort(Nmax)[::-1]
    yo, xo = yo[isort], xo[isort]
    pix = []
    for t in range(len(yo)):
        pix.append([yo[t],xo[t]])

    for iter in range(5):
        for k in range(len(pix)):
            ye, xe = extendROI(pix[k][0], pix[k][1], h.shape[0], h.shape[1], 1)
            igood = h[ye, xe]>2
            ye, xe = ye[igood], xe[igood]
            pix[k][0] = ye
            pix[k][1] = xe

    ibad = np.ones(len(pix), 'bool')
    for k in range(len(pix)):
        if pix[k][0].size<nmax:
            ibad[k] = 0

    #pix = [pix[k] for k in ibad.nonzero()[0]]

    M = np.zeros(h.shape)
    for k in range(len(pix)):
        M[pix[k][0],    pix[k][1]] = 1+k

    M0 = M[rpad + xp, rpad + yp]
    M0 = np.reshape(M0, xm.shape)
    return M0, pix

def normalize99(img):
    X = img.copy()
    X = (X - np.percentile(X, 1)) / (np.percentile(X, 99) - np.percentile(X, 1))
    return X

def reshape(data, channels=[0,0]):
    data = data.astype(np.float32)
    # use grayscale image
    if channels[0]==0:
        data = data.mean(axis=-1)
        data = np.expand_dims(data, axis=-1)
    else:
        chanid = [channels[0]-1]
        if channels[1] > 0:
            chanid.append(channels[1]-1)
            data = data[:,:,chanid]
        else:
            data = data[:,:,chanid[0]]
            data = np.expand_dims(data, axis=-1)
    if len(channels)>1 and data.shape[-1]==1:
        data = np.concatenate((data, np.zeros_like(data)), axis=-1)
    elif len(channels)==1 and data.shape[-1]>1:
        data = data[...,:-1]
    if data.ndim > 3:
        data = np.transpose(data, (3,0,1,2))
    else:
        data = np.transpose(data, (2,0,1))
    return data


def pad_image_CS(img0, div=16, extra = 1):
    Lpad = int(div * np.ceil(img0.shape[-2]/div) - img0.shape[-2])
    xpad1 = extra*div//2 + Lpad//2
    xpad2 = extra*div//2 + Lpad - Lpad//2
    Lpad = int(div * np.ceil(img0.shape[-1]/div) - img0.shape[-1])
    ypad1 = extra*div//2 + Lpad//2
    ypad2 = extra*div//2+Lpad - Lpad//2

    if img0.ndim>3:
        pads = np.array([[0,0], [0,0], [xpad1,xpad2], [ypad1, ypad2]])
    else:
        pads = np.array([[0,0], [xpad1,xpad2], [ypad1, ypad2]])

    I = np.pad(img0,pads, mode='constant')
    return I,pads

def run_dynamics(y):
    x0, y0 = np.meshgrid(np.arange(y.shape[-1]),  np.arange(y.shape[-2]))

    xs, ys = x0.copy(), y0.copy()
    yout = np.zeros(y.shape)
    Ly, Lx = y[0,0].shape
    for k in range(y.shape[0]):
        dx = y[k,0,:,:]
        dy = y[k,1,:,:]
        for j in range(200):
            xi = xs.astype('int')
            yi = ys.astype('int')
            xi = np.clip(xi, 0, Lx-1)
            yi = np.clip(yi, 0, Ly-1)

            xs = np.clip(xs - .1*dx[yi, xi], 0, Lx-1)
            ys = np.clip(ys - .1*dy[yi, xi], 0, Ly-1)
        yout[k,0] = ys
        yout[k,1] = xs
    return yout


def batchconv(nconv, sz):
    conv = nn.HybridSequential()
    with conv.name_scope():
        conv.add(
                nn.Conv2D(nconv, kernel_size=sz, padding=sz // 2),
                nn.BatchNorm(axis=1),
                nn.Activation('relu'),
        )
    return conv

def batchdense(nfeat):
    conv = nn.HybridSequential()
    with conv.name_scope():
        conv.add(
                nn.Dense(nfeat),
                #nn.BatchNorm(axis=1),
                nn.Activation('relu'),
        )
    return conv

def downblock(nconv, sz, sz2, pool=True):
    conv = nn.HybridSequential()
    if pool:
        conv.add(nn.AvgPool2D(pool_size=(2,2), strides=(2,2)))
    conv.add(batchconv(nconv, sz))
    conv.add(batchconv(nconv, sz2))
    return conv

class densedownsample(nn.HybridBlock):
    def __init__(self, nbase2, **kwargs):
        super(densedownsample, self).__init__(**kwargs)
        with self.name_scope():
            self.down = nn.HybridSequential()
            self.pool = nn.AvgPool2D(pool_size=(2,2), strides=(2,2))
            for n in range(len(nbase2)):
                self.down.add(batchconv(nbase2[n], 3))

    def hybrid_forward(self, F, x):
        xd = self.pool(x)
        for n in range(len(self.down)):
            y = self.down[n](xd)
            xd = F.concat(xd, y, dim=1)
        return y


class downsample(nn.HybridBlock):
    def __init__(self, nbase, sz, sz2, **kwargs):
        super(downsample, self).__init__(**kwargs)
        with self.name_scope():
            self.down = nn.HybridSequential()
            for n in range(len(nbase)):
                if n==1:
                    self.down.add(densedownsample([16, 32, 64, 128, 256]))
                else:
                    self.down.add(downblock(nbase[n], sz[n], sz2[n], n>0))


    def hybrid_forward(self, F, x):
        xd = [self.down[0](x)]
        for n in range(1, len(self.down)):
            xd.append(self.down[n](xd[n-1]))
        return xd

class upblock(nn.HybridBlock):
    def __init__(self, nconv, sz, **kwargs):
        super(upblock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv0 = batchconv(nconv, sz)
            self.conv1 = batchconv(nconv, sz)
            self.full = nn.Dense(nconv)

    def hybrid_forward(self, F, y, x, style):
        y = self.conv0(y)
        y = x + y #F.concat(x, y, dim=1)
        y = self.conv1(y)
        y = F.broadcast_add(y , self.full(style).expand_dims(-1).expand_dims(-1))
        y = F.relu(y)
        return y

class upsample(nn.HybridBlock):
    def __init__(self, nbase, sz, **kwargs):
        super(upsample, self).__init__(**kwargs)
        with self.name_scope():
            self.up = nn.HybridSequential()
            for n in range(len(nbase)):
                self.up.add(upblock(nbase[n], sz[n]))

    def hybrid_forward(self, F, style, xd):#x0, x1, x2, x3, style):
        y = self.up[-1](xd[-1], xd[-1], style)
        for n in range(len(self.up)-2,-1,-1):
            y = F.UpSampling(y, scale=2, sample_type='nearest')
            y = self.up[n](y, xd[n], style)
        return y

class output(nn.HybridBlock):
    def __init__(self, nfeat, szf, **kwargs):
        super(output, self).__init__(**kwargs)
        with self.name_scope():
            self.full0    = nn.Dense(nfeat)
            self.uconv0   = nn.Conv2D(channels=nfeat,      kernel_size=szf[0],  padding = szf[0]//2)
            self.ubatchnorm0 = nn.BatchNorm(axis=1)
            self.oconv1   = nn.Conv2D(channels=3,    kernel_size=szf[0],  padding = szf[0]//2)

    def hybrid_forward(self, F, y0, style):
        y = self.ubatchnorm0(self.uconv0(y0))
        feat = self.full0(style)
        y = F.broadcast_add(y, feat.expand_dims(-1).expand_dims(-1))
        y = F.relu(y)
        y = self.oconv1(y)
        return y

class make_style(nn.HybridBlock):
    def __init__(self, nbase, **kwargs):
        super(make_style, self).__init__(**kwargs)
        with self.name_scope():
            self.pool_all = nn.GlobalAvgPool2D()
            #self.flatten = nn.Flatten()
            #self.full = nn.HybridSequential()
            #for j in range(len(nbase)):
        #        self.full.add(batchdense(nbase[j]))

    def hybrid_forward(self, F, x0):
        style = self.pool_all(x0)
        svar  = self.pool_all(x0**2)
        #style = self.flatten(style)
        style = F.broadcast_div(style, F.sum(style**2, axis=1).expand_dims(1)**.5)
        svar  = F.broadcast_div(svar,  F.sum(svar**2, axis=1).expand_dims(1)**.5)

        #for j in range(len(self.full)):
        #    y = self.full[j](style)
        #    style = F.concat(style, y, dim=1)
        return style, svar

class CPnet(gluon.HybridBlock):

    def __init__(self, nbase, **kwargs):
        super(CPnet, self).__init__(**kwargs)
        with self.name_scope():
            self.downsample = downsample(nbase, sz, sz2)
            self.upsample = upsample(nbase, sz)
            self.output = output(nfeat, szf)
            self.make_style = make_style([64, 128, 256])

    def hybrid_forward(self, F, data):
        xd    = self.downsample(data)
        style, svar = self.make_style(xd[-1])
        y0    = self.upsample(style, xd)
        T0    = self.output(y0, style)
        style = F.concat(style, svar, dim=1)
        return T0, style

class CellposeModel():
    def __init__(self, device, pretrained_model=None, **kwargs):
        super(CellposeModel, self).__init__(**kwargs)
        self.device = device
        self.pretrained_model = pretrained_model
        nbase = [16,256,128,64]
        self.net = CPnet(nbase)
        self.net.hybridize()
        self.net.initialize(ctx = self.device)
        if pretrained_model is not None:
            self.net.load_parameters(pretrained_model)

    def eval(self, data, channels=[0,0], do_3D=False):
        batch_size=8
        data = reshape(data, channels=channels)
        if self.pretrained_model=='nuclei':
            data = data[:1]
        # rescale image
        x = normalize99(data)
        Ly,Lx = x.shape[-2:]
        while x.ndim<4:
            x = np.expand_dims(x, 0)
        nimg = x.shape[0]
        flows = [[],[],[],[]]
        x, pads = pad_image_CS(x)
        for ibatch in range(0,nimg,batch_size):
            X = nd.array(x[ibatch:ibatch+batch_size], ctx=self.device)
            # run network
            y = self.net(X)[0]
            y = y.detach().asnumpy()
            # undo padding
            y = y[:, :, pads[-2][0]:y.shape[-2]-pads[-2][-1], pads[-1][0]:y.shape[-1]-pads[-1][-1]]
            # compute dynamics from flows
            for k in range(y.shape[0]):
                yout = run_dynamics((-y[k,:2] * (y[k,2]>0.))[np.newaxis,...])
                masks = get_mask(yout[0])[0]
                y[k,0] /= np.abs(y[k,0]).max()
                y[k,1] /= np.abs(y[k,1]).max()
                flows[0].append(y[k,0])#(y[k,0]/5 * 127 + 127).astype(np.uint8))
                flows[1].append(y[k,1])#(y[k,1]/5 * 127 + 127).astype(np.uint8))
                flows[2].append(np.zeros((Ly,Lx), np.uint8))
                flows[3].append(np.clip(y[k,-1] * 127 + 127, 0, 255).astype(np.uint8))
        return masks, flows

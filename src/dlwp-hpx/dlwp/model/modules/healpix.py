#! env/bin/python3

"""
This file contains padding and convolution classes to perform according operations on the twelve faces of the HEALPix.


         HEALPix                              Face order                 3D array representation
                                                                            -----------------
--------------------------               //\\  //\\  //\\  //\\             |   |   |   |   |
|| 0  |  1  |  2  |  3  ||              //  \\//  \\//  \\//  \\            |0  |1  |2  |3  |
|\\  //\\  //\\  //\\  //|             /\\0 //\\1 //\\2 //\\3 //            -----------------
| \\//  \\//  \\//  \\// |            // \\//  \\//  \\//  \\//             |   |   |   |   |
|4//\\5 //\\6 //\\7 //\\4|            \\4//\\5 //\\6 //\\7 //\\             |4  |5  |6  |7  |
|//  \\//  \\//  \\//  \\|             \\/  \\//  \\//  \\//  \\            -----------------
|| 8  |  9  |  10 |  11  |              \\8 //\\9 //\\10//\\11//            |   |   |   |   |
--------------------------               \\//  \\//  \\//  \\//             |8  |9  |10 |11 |
                                                                            -----------------
                                    "\\" are top and bottom, whereas
                                    "//" are left and right borders


Details on the HEALPix can be found at https://iopscience.iop.org/article/10.1086/427976

"""

import numpy as np
import torch
import torch as th
from einops.layers.torch import Rearrange

have_healpixpad = False
try:
    from healpixpad import HEALPixPad
    have_healpixpad = True
except ImportError:
    print("Warning, cannot find healpixpad module")
    have_healpixpad = False


# converts an NFCHW tensor to an NFHWC tensor
@torch.jit.script
def healpix_channels_first_to_channels_last(tensor):
    N, F, C, H, W = tensor.shape
    tensor = torch.permute(tensor, dims=(0, 1, 3, 4, 2))
    tensor = torch.as_strided(tensor,
                              size=(N, F, C, H, W),
                              stride=(F*H*W*C, H*W*C, 1, W*C, C))
    
    return tensor

# perform face folding:
# [B, F, C, H, W] -> [B*F, C, H, W]
class HEALPixFoldFaces(th.nn.Module):

    def __init__(self, enable_nhwc = False):
        super().__init__()
        self.enable_nhwc = enable_nhwc

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        
        N, F, C, H, W = tensor.shape
        tensor = torch.reshape(tensor, shape=(N*F, C, H, W))

        if self.enable_nhwc:
            tensor = tensor.to(memory_format=torch.channels_last)
    
        return tensor


class HEALPixUnfoldFaces(th.nn.Module):

    def __init__(self, num_faces=12, enable_nhwc = False):
        super().__init__()
        self.num_faces = num_faces
        self.enable_nhwc = enable_nhwc

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        
        NF, C, H, W = tensor.shape
        tensor = torch.reshape(tensor, shape=(-1, self.num_faces, C, H, W))
    
        return tensor


class HEALPixLayer(th.nn.Module):
    """
    Pytorch module for applying any base torch Module on a HEALPix tensor. Expects all input/output tensors to have a
    shape [..., 12, H, W], where 12 is the dimension of the faces.
    """
    def __init__(self, layer, **kwargs):
        """
        Constructor for the HEALPix base layer.

        :param layer: Any torch layer function, e.g., th.nn.Conv2d
        :param kwargs: The arguments that are passed to the torch layer function, e.g., kernel_size
        """
        super().__init__()
        layers = []

        # If 'layer' is a string, convert into the according function
        if isinstance(layer, str): layer = eval(layer)

        enable_nhwc = kwargs["enable_nhwc"]
        del kwargs["enable_nhwc"]

        enable_healpixpad = kwargs["enable_healpixpad"]
        del kwargs["enable_healpixpad"] 
        
        # Define a HEALPixPadding layer if the given layer is a convolution layer
        try:
            if layer.__bases__[0] is th.nn.modules.conv._ConvNd and kwargs["kernel_size"] > 1:
                kwargs["padding"] = 0  # Disable native padding
                kernel_size = 3 if "kernel_size" not in kwargs else kwargs["kernel_size"]
                dilation = 1 if "dilation" not in kwargs else kwargs["dilation"]
                padding = ((kernel_size - 1)//2)*dilation
                if enable_healpixpad and have_healpixpad and th.cuda.is_available() and not enable_nhwc:
                    layers.append(HEALPixPaddingv2(padding=padding))
                else:
                    layers.append(HEALPixPadding(padding=padding, enable_nhwc=enable_nhwc))
        except AttributeError:
            print(f"Could not determine the base class of the given layer '{layer}'. No padding layer was added, "
                   "which may not be an issue if the specified layer does not require a previous padding.")
            
        # Initialize the desired pytorch layer surrounded by tensor reshaping functions that enable the layer to
        # process all faces in parallel on the batch dimension
        #layers.append(Rearrange("b f ... h w -> (b f) ... h w"))
        #layers.append(FoldFaces())
        layers.append(layer(**kwargs))
        #layers.append(Rearrange("(b f) ... h w -> b f ... h w", f=12))  # Inverts face-to-batch reshape from above
        #layers.append(UnfoldFaces(num_faces=12)) 
        self.layers = th.nn.Sequential(*layers)

        if enable_nhwc:
            self.layers = self.layers.to(memory_format=torch.channels_last)

    def forward(self, x: th.Tensor,) -> th.Tensor:
        """
        Performs the forward pass using the defined layer function and the given data.

        :param x: The input tensor of shape [..., F=12, H, W]
        :return: The output tensor of this HEALPix layer
        """
        res = self.layers(x)
        return res


class HEALPixPaddingv2(th.nn.Module):
    def __init__(self, padding: int):
        super().__init__()
        self.unfold = HEALPixUnfoldFaces(num_faces=12)
        self.fold = HEALPixFoldFaces()
        self.padding = HEALPixPad(padding=padding)

    def forward(self, x):
        torch.cuda.nvtx.range_push("HEALPixPaddingv2:forward")
        
        x = self.unfold(x)
        xp = self.padding(x)
        xp = self.fold(xp)
        
        torch.cuda.nvtx.range_pop()

        return xp


class HEALPixPadding(th.nn.Module):
    """
    Padding layer for data on a HEALPix sphere. The requirements for using this layer are as follows:
    - The last three dimensions are (face=12, height, width)
    - The first four indices in the faces dimension [0, 1, 2, 3] are the faces on the northern hemisphere
    - The second four indices in the faces dimension [4, 5, 6, 7] are the faces on the equator
    - The last four indices in the faces dimension [8, 9, 10, 11] are the faces on the southern hemisphere

    Orientation and arrangement of the HEALPix faces are outlined above.
    """

    def __init__(self, padding: int, enable_nhwc: bool = False):
        """
        Constructor for a HEALPix padding layer.

        :param padding: The padding size
        """
        super().__init__()
        self.p = padding
        self.d = [-2, -1]
        self.ret_tl = th.zeros(1, 1, 1)
        self.ret_br = th.zeros(1, 1, 1)
        self.enable_nhwc = enable_nhwc
        if not isinstance(padding, int) or padding < 1:
            raise ValueError(f"invalid value for 'padding', expected int > 0 but got {padding}")

        self.fold = HEALPixFoldFaces(enable_nhwc = self.enable_nhwc)
        self.unfold = HEALPixUnfoldFaces(num_faces = 12, enable_nhwc = self.enable_nhwc)

    def forward(self, data: th.Tensor) -> th.Tensor:
        """
        Pad each face consistently with its according neighbors in the HEALPix (see ordering and neighborhoods above).

        :param data: The input tensor of shape [..., F, H, W] where each face is to be padded in its HPX context
        :return: The padded tensor where each face's height and width are increased by 2*p
        """
        torch.cuda.nvtx.range_push("HEALPixPadding:forward")
        
        # unfold faces from batch dim
        data = self.unfold(data)
        
        # Extract the twelve faces (as views of the original tensors)
        f00, f01, f02, f03, f04, f05, f06, f07, f08, f09, f10, f11 = [torch.squeeze(x, dim=1) for x in th.split(tensor=data, split_size_or_sections=1, dim=1)]
        
        # Assemble the four padded faces on the northern hemisphere
        p00 = self.pn(c=f00, t=f01, tl=f02, l=f03, bl=f03, b=f04, br=f08, r=f05, tr=f01)
        p01 = self.pn(c=f01, t=f02, tl=f03, l=f00, bl=f00, b=f05, br=f09, r=f06, tr=f02)
        p02 = self.pn(c=f02, t=f03, tl=f00, l=f01, bl=f01, b=f06, br=f10, r=f07, tr=f03)
        p03 = self.pn(c=f03, t=f00, tl=f01, l=f02, bl=f02, b=f07, br=f11, r=f04, tr=f00)
        
        # Assemble the four padded faces on the equator
        p04 = self.pe(c=f04, t=f00, tl=self.tl(f00, f03), l=f03, bl=f07, b=f11, br=self.br(f11, f08), r=f08, tr=f05)
        p05 = self.pe(c=f05, t=f01, tl=self.tl(f01, f00), l=f00, bl=f04, b=f08, br=self.br(f08, f09), r=f09, tr=f06)
        p06 = self.pe(c=f06, t=f02, tl=self.tl(f02, f01), l=f01, bl=f05, b=f09, br=self.br(f09, f10), r=f10, tr=f07)
        p07 = self.pe(c=f07, t=f03, tl=self.tl(f03, f02), l=f02, bl=f06, b=f10, br=self.br(f10, f11), r=f11, tr=f04)
        
        # Assemble the four padded faces on the southern hemisphere
        p08 = self.ps(c=f08, t=f05, tl=f00, l=f04, bl=f11, b=f11, br=f10, r=f09, tr=f09)
        p09 = self.ps(c=f09, t=f06, tl=f01, l=f05, bl=f08, b=f08, br=f11, r=f10, tr=f10)
        p10 = self.ps(c=f10, t=f07, tl=f02, l=f06, bl=f09, b=f09, br=f08, r=f11, tr=f11)
        p11 = self.ps(c=f11, t=f04, tl=f03, l=f07, bl=f10, b=f10, br=f09, r=f08, tr=f08)

        res = th.stack((p00, p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11), dim=1)

        # fold faces into batch dim
        res = self.fold(res)
            
        torch.cuda.nvtx.range_pop()
                
        return res

    def pn(self, c: th.Tensor, t: th.Tensor, tl: th.Tensor, l: th.Tensor, bl: th.Tensor, b: th.Tensor, br: th.Tensor,
           r: th.Tensor, tr: th.Tensor) -> th.Tensor:
        """
        Applies padding to a northern hemisphere face c under consideration of its given neighbors.

        :param c: The central face and tensor that is subject for padding
        :param t: The top neighboring face tensor
        :param tl: The top left neighboring face tensor
        :param l: The left neighboring face tensor
        :param bl: The bottom left neighboring face tensor
        :param b: The bottom neighboring face tensor
        :param br: The bottom right neighboring face tensor
        :param r: The right neighboring face tensor
        :param tr: The top right neighboring face  tensor
        :return: The padded tensor p
        """
        p = self.p  # Padding size
        d = self.d  # Dimensions for rotations

        # Start with top and bottom to extend the height of the c tensor
        c = th.cat((t.rot90(1, d)[..., -p:, :], c, b[..., :p, :]), dim=-2)

        # Construct the left and right pads including the corner faces
        left = th.cat((tl.rot90(2, d)[..., -p:, -p:], l.rot90(-1, d)[..., -p:], bl[..., :p, -p:]), dim=-2)
        right = th.cat((tr[..., -p:, :p], r[..., :p], br[..., :p, :p]), dim=-2)

        return th.cat((left, c, right), dim=-1)

    def pe(self, c: th.Tensor, t: th.Tensor, tl: th.Tensor, l: th.Tensor, bl: th.Tensor, b: th.Tensor, br: th.Tensor,
           r: th.Tensor, tr: th.Tensor) -> th.Tensor:
        """
        Applies padding to an equatorial face c under consideration of its given neighbors.

        :param c: The central face and tensor that is subject for padding
        :param t: The top neighboring face tensor
        :param tl: The top left neighboring face tensor
        :param l: The left neighboring face tensor
        :param bl: The bottom left neighboring face tensor
        :param b: The bottom neighboring face tensor
        :param br: The bottom right neighboring face tensor
        :param r: The right neighboring face tensor
        :param tr: The top right neighboring face  tensor
        :return: The padded tensor p
        """
        p = self.p  # Padding size
        d = self.d  # Dimensions for rotations

        # Start with top and bottom to extend the height of the c tensor
        c = th.cat((t[..., -p:, :], c, b[..., :p, :]), dim=-2)

        # Construct the left and right pads including the corner faces
        left = th.cat((tl[..., -p:, -p:], l[..., -p:], bl[..., :p, -p:]), dim=-2)
        right = th.cat((tr[..., -p:, :p], r[..., :p], br[..., :p, :p]), dim=-2)

        return th.cat((left, c, right), dim=-1)

    def ps(self, c: th.Tensor, t: th.Tensor, tl: th.Tensor, l: th.Tensor, bl: th.Tensor, b: th.Tensor, br: th.Tensor,
           r: th.Tensor, tr: th.Tensor) -> th.Tensor:
        """
        Applies padding to a southern hemisphere face c under consideration of its given neighbors.

        :param c: The central face and tensor that is subject for padding
        :param t: The top neighboring face tensor
        :param tl: The top left neighboring face tensor
        :param l: The left neighboring face tensor
        :param bl: The bottom left neighboring face tensor
        :param b: The bottom neighboring face tensor
        :param br: The bottom right neighboring face tensor
        :param r: The right neighboring face tensor
        :param tr: The top right neighboring face  tensor
        :return: The padded tensor p
        """
        p = self.p  # Padding size
        d = self.d  # Dimensions for rotations

        # Start with top and bottom to extend the height of the c tensor
        c = th.cat((t[..., -p:, :], c, b.rot90(1, d)[..., :p, :]), dim=-2)

        # Construct the left and right pads including the corner faces
        left = th.cat((tl[..., -p:, -p:], l[..., -p:], bl[..., :p, -p:]), dim=-2)
        right = th.cat((tr[..., -p:, :p], r.rot90(-1, d)[..., :p], br.rot90(2, d)[..., :p, :p]), dim=-2)
        
        return th.cat((left, c, right), dim=-1)
        
    def tl(self, t: th.Tensor, l: th.Tensor) -> th.Tensor:
        """
        Assembles the top left corner of a center face in the cases where no according top left face is defined on the
        HPX.

        :param t: The face above the center face
        :param l: The face left of the center face
        :return: The assembled top left corner (only the sub-part that is required for padding)
        """
        #ret = th.zeros((*t.shape[:-2], self.p, self.p), dtype=t.dtype, device=t.device)
        ret = th.zeros_like(t)[..., :self.p, :self.p]  # super ugly but super fast
        #tc = t[..., :self.p, :self.p]
        #lc = l[..., :self.p, :self.p]
        #td = torch.diag_embed(torch.diagonal(tc, dim1=-2, dim2=-1, offset=1), dim1=-2, dim2=-1)
        #ld = torch.diag_embed(torch.diagonal(lc, dim1=-2, dim2=-1, offset=-1), dim1=-2, dim2=-1) 
        #ret2 = torch.triu(tc, diagonal = 1) + torch.tril(lc, diagonal = -1) + 0.5 * (td + ld)

        # Bottom left point
        ret[..., -1, -1] = 0.5*t[..., -1, 0] + 0.5*l[..., 0, -1]
    
        # Remaining points
        for i in range(1, self.p):
            ret[..., -i-1, -i:] = t[..., -i-1, :i]  # Filling top right above main diagonal
            ret[..., -i:, -i-1] = l[..., :i, -i-1]  # Filling bottom left below main diagonal
            ret[..., -i-1, -i-1] = 0.5*t[..., -i-1, 0] + 0.5*l[..., 0, -i-1]  # Diagonal

        #print("ALL GOOD", torch.allclose(ret, ret2), ret[0,0,...], ret2[0,0,...])
        #sys.exit(1)
        
        return ret

    def br(self, b: th.Tensor, r: th.Tensor) -> th.Tensor:
        """
        Assembles the bottom right corner of a center face in the cases where no according bottom right face is defined
        on the HPX.

        :param b: The face below the center face
        :param r: The face right of the center face
        :return: The assembled bottom right corner (only the sub-part that is required for padding)
        """
        #ret = th.zeros((*b.shape[:-2], self.p, self.p), dtype=b.dtype, device=b.device)
        ret = th.zeros_like(b)[..., :self.p, :self.p]

        # Top left point
        ret[..., 0, 0] = 0.5*b[..., 0, -1] + 0.5*r[..., -1, 0]

        # Remaining points
        for i in range(1, self.p):
            ret[..., :i, i] = r[..., -i:, i]  # Filling top right above main diagonal
            ret[..., i, :i] = b[..., i, -i:]  # Filling bottom left below main diagonal
            ret[..., i, i] = 0.5*b[..., i, -1] + 0.5*r[..., -1, i]  # Diagonal

        return ret

def visualize_healpix(data: np.array, s: int = 1e12, **kwargs):
    """
    Visualizes HEALPix data that are stored in a rectangular data structure.

    :param data: The data for visualization in shape [f, h, w] (faces=12, height, width)
    :param s: (Optional) A scalar used for masking the data
    :param **kwargs: (Optional) Additional plotting parameters for imshow (e.g., vmin, vmax)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import rotate

    if "vmin" not in kwargs or "vmax" not in kwargs:
        kwargs["vmin"], kwargs["vmax"] = data.min(), data.max()

    # Concatenate the faces in a HEALPix-like diamond structure
    f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11 = data

    nans = np.ones_like(f0)*s
    row0 = np.concatenate((nans, nans, nans, f3, nans), axis=1)
    row1 = np.concatenate((nans, nans, f2, f7, f11), axis=1)
    row2 = np.concatenate((nans, f1, f6, f10, nans), axis=1)
    row3 = np.concatenate((f0, f5, f9, nans, nans), axis=1)
    row4 = np.concatenate((f4, f8, nans, nans, nans), axis=1)
    data = np.concatenate((row0, row1, row2, row3, row4), axis=0)

    # Create mask and set all masked data points to zero (necessary for successfull rotation)
    mask = np.ones_like(data, dtype=np.int32)*(-s)
    mask[data==s] = s
    data[mask==s] = 0.0

    # Rotate data and mask and apply mask to rotated data
    data = rotate(data, angle=-45, reshape=True)
    mask = rotate(mask, angle=-45, reshape=True)
    mask[mask==0.0] = s
    data[mask > s/2] = np.nan

    # Crop and plot
    h, w = data.shape
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.imshow(data[int(h/3.3):h-int(h/3.3), :int(w*0.91)], **kwargs)
    ax.set_title("(Border artifacts caused by rotation)")
    plt.tight_layout()
    #plt.show()


if __name__ == "__main__":
    # For debugging purposes

    data = th.randn(1, 12, 1, 32, 32)
    print(data.shape)
    
    #import xarray as xr
    #data[0, 0] = th.tensor(xr.open_dataset("era5_1deg_3h_HPX32_1979-2022_land_sea_mask.nc")["lsm"].values)

    # 2D HEALPix convolution layer (hard coded example)
    layer = HEALPixLayer(layer=th.nn.Conv2d, in_channels=1, out_channels=1, kernel_size=3, bias=False)
    print(layer)

    # Set all convolution weights to 1/9 to realize a smoothing of the input
    layer.layers[2].weight = th.nn.Parameter(th.tensor([[[[1/9, 1/9, 1/9],
                                                          [1/9, 1/9, 1/9],
                                                          [1/9, 1/9, 1/9]]]]).type_as(layer.layers[2].weight))
    x = layer(x=data)
    visualize_healpix(data=x[0, 0].detach().cpu().numpy())

    # Padding alone
    padding = HEALPixPadding(padding=1)
    data = padding(data=data)
    visualize_healpix(data=data[0, 0].numpy())

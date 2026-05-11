import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


def soft_erode(img: torch.Tensor) -> torch.Tensor:  # type: ignore
    """
    Perform soft erosion on the input image

    Args:
        img: the shape should be BCH(WD)

    Adapted from:
        https://github.com/jocpae/clDice/blob/master/cldice_loss/pytorch/soft_skeleton.py#L6
    """
    if len(img.shape) == 4:
        p1 = -(F.max_pool2d(-img, (3, 1), (1, 1), (1, 0)))
        p2 = -(F.max_pool2d(-img, (1, 3), (1, 1), (0, 1)))
        return torch.min(p1, p2)
    elif len(img.shape) == 5:
        p1 = -(F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0)))
        p2 = -(F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0)))
        p3 = -(F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1)))
        return torch.min(torch.min(p1, p2), p3)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:  # type: ignore
    """
    Perform soft dilation on the input image

    Args:
        img: the shape should be BCH(WD)

    Adapted from:
        https://github.com/jocpae/clDice/blob/master/cldice_loss/pytorch/soft_skeleton.py#L18
    """
    if len(img.shape) == 4:
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
    elif len(img.shape) == 5:
        return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))


def soft_open(img: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to perform soft opening on the input image

    Args:
        img: the shape should be BCH(WD)

    Adapted from:
        https://github.com/jocpae/clDice/blob/master/cldice_loss/pytorch/soft_skeleton.py#L25
    """
    eroded_image = soft_erode(img)
    dilated_image = soft_dilate(eroded_image)
    return dilated_image


def soft_skel(img: torch.Tensor, iter_: int) -> torch.Tensor:
    """
    Perform soft skeletonization on the input image

    Adapted from:
       https://github.com/jocpae/clDice/blob/master/cldice_loss/pytorch/soft_skeleton.py#L29

    Args:
        img: the shape should be BCH(WD)
        iter_: number of iterations for skeletonization

    Returns:
        skeletonized image
    """
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


class SoftclDiceLoss(_Loss):
    """
    Compute the Soft clDice loss defined in:

        Shit et al. (2021) clDice -- A Novel Topology-Preserving Loss Function
        for Tubular Structure Segmentation. (https://arxiv.org/abs/2003.07311)

    Adapted from:
        https://github.com/jocpae/clDice/blob/master/cldice_loss/pytorch/cldice.py#L7
    """

    def __init__(self, iter_: int = 3, smooth: float = 1.0, sigmoid=False) -> None:
        """
        Args:
            iter_: Number of iterations for skeletonization
            smooth: Smoothing parameter
        """
        super().__init__()
        self.iter = iter_
        self.smooth = smooth
        self.sigmoid = sigmoid

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.sigmoid:
            y_pred = y_pred.sigmoid()
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        if y_pred.shape[1] == 1:
            tprec = (torch.sum(torch.multiply(skel_pred, y_true)) + self.smooth) / (
                torch.sum(skel_pred) + self.smooth
            )
            tsens = (torch.sum(torch.multiply(skel_true, y_pred)) + self.smooth) / (
                torch.sum(skel_true) + self.smooth
            )
        else:
            tprec = (
                torch.sum(torch.multiply(skel_pred, y_true)[:, 1:, ...]) + self.smooth
            ) / (torch.sum(skel_pred[:, 1:, ...]) + self.smooth)
            tsens = (
                torch.sum(torch.multiply(skel_true, y_pred)[:, 1:, ...]) + self.smooth
            ) / (torch.sum(skel_true[:, 1:, ...]) + self.smooth)
        cl_dice: torch.Tensor = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)
        return cl_dice
    

def soft_dice(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    smooth: float = 1e-5,
    squared_pred: bool = False,
) -> torch.Tensor:
    """
    Soft Dice loss with optional squared_pred.

    Args:
        y_true: (B, C, ...)
        y_pred: (B, C ...)
        smooth: small epsilon for numerical stability
        squared_pred: whether to square predictions (and targets) in denominator

    Returns:
        dice loss (scalar)
    """
    if y_pred.shape[1] == 1:
        intersection = torch.sum(y_true * y_pred)

        if squared_pred:
            denom = torch.sum(y_true * y_true) + torch.sum(y_pred * y_pred)
        else:
            denom = torch.sum(y_true) + torch.sum(y_pred)

        coeff = (2.0 * intersection + smooth) / (denom + smooth)
    else:
        # exclude background channel 0
        intersection = torch.sum((y_true * y_pred)[:, 1:, ...])

        if squared_pred:
            denom = (
                torch.sum((y_true * y_true)[:, 1:, ...])
                + torch.sum((y_pred * y_pred)[:, 1:, ...])
            )
        else:
            denom = (
                torch.sum(y_true[:, 1:, ...])
                + torch.sum(y_pred[:, 1:, ...])
            )
        coeff = (2.0 * intersection + smooth) / (denom + smooth)
    return 1.0 - coeff


class SoftDiceclDiceLoss(_Loss):
    """
    Compute the Soft clDice loss defined in:

        Shit et al. (2021) clDice -- A Novel Topology-Preserving Loss Function
        for Tubular Structure Segmentation. (https://arxiv.org/abs/2003.07311)

    Adapted from:
        https://github.com/jocpae/clDice/blob/master/cldice_loss/pytorch/cldice.py#L38
    """

    def __init__(self, iter_: int = 3, alpha: float = 0.5, smooth: float = 1.0, sigmoid=False, squared_pred=False, clDice_start_step=0, allow_delay_start=True) -> None:
        """
        Args:
            iter_: Number of iterations for skeletonization
            smooth: Smoothing parameter
            alpha: Weighing factor for cldice
        """
        super().__init__()
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha
        self.sigmoid = sigmoid
        self.squared_pred = squared_pred
        self.clDice_start_step = clDice_start_step
        self.allow_delay_start = allow_delay_start

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, global_step = None) -> torch.Tensor:
        if self.sigmoid:
            y_pred = y_pred.sigmoid()

        dice = soft_dice(y_true, y_pred, self.smooth, squared_pred=self.squared_pred)
        if global_step is not None and global_step < self.clDice_start_step:
            return dice
        
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        if y_pred.shape[1] == 1:
            tprec = (torch.sum(torch.multiply(skel_pred, y_true)) + self.smooth) / (
                torch.sum(skel_pred) + self.smooth
            )
            tsens = (torch.sum(torch.multiply(skel_true, y_pred)) + self.smooth) / (
                torch.sum(skel_true) + self.smooth
            )
        else:
            tprec = (torch.sum(torch.multiply(skel_pred, y_true)[:, 1:, ...]) + self.smooth) / (
                torch.sum(skel_pred[:, 1:, ...]) + self.smooth
            )
            tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:, 1:, ...]) + self.smooth) / (
                torch.sum(skel_true[:, 1:, ...]) + self.smooth
            )
        cl_dice = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)
        total_loss: torch.Tensor = (1.0 - self.alpha) * dice + self.alpha * cl_dice
        return total_loss


import torch
from torch import nn


class MK_MMD_Loss(nn.Module):
    def __init__(self, kernel_alphas=None):
        """
        Multiple Kernel Maximum Mean Discrepancy (MK-MMD).
        kernel_alphas: bandwidths for Gaussian RBF kernels.
        """
        super(MK_MMD_Loss, self).__init__()
        if kernel_alphas is None:
            kernel_alphas = [0.5, 1.0, 2.0, 5.0, 10.0]
        self.kernel_alphas = kernel_alphas

    def compute_rbf_kernel(self, x, y, alpha):
        """Compute Gaussian RBF kernel matrix between rows of x and y."""
        x_expand = x.unsqueeze(1)
        y_expand = y.unsqueeze(0)
        l2_dist_sq = torch.pow(x_expand - y_expand, 2).sum(dim=2)
        return torch.exp(-l2_dist_sq / alpha)

    @staticmethod
    def _off_diagonal_mean(kernel_mat):
        n = kernel_mat.size(0)
        if n <= 1:
            return kernel_mat.new_tensor(0.0)
        return (kernel_mat.sum() - torch.trace(kernel_mat)) / (n * (n - 1))

    def forward(self, z_source, z_target):
        """
        z_source: [N, D] source latent vectors.
        z_target: [M, D] target latent vectors.
        """
        mmd_loss = z_source.new_tensor(0.0)

        for alpha in self.kernel_alphas:
            xx = self.compute_rbf_kernel(z_source, z_source, alpha)
            yy = self.compute_rbf_kernel(z_target, z_target, alpha)
            xy = self.compute_rbf_kernel(z_source, z_target, alpha)

            xx_mean = self._off_diagonal_mean(xx)
            yy_mean = self._off_diagonal_mean(yy)
            xy_mean = torch.mean(xy)
            mmd_loss = mmd_loss + (xx_mean + yy_mean - 2.0 * xy_mean)

        return mmd_loss


def calculate_mmd_distance(z_source, z_target, kernel_alphas=None):
    """
    Compute MK-MMD as a Python float for evaluation only.
    """
    if z_target.device != z_source.device:
        z_target = z_target.to(z_source.device)

    mmd_loss = MK_MMD_Loss(kernel_alphas=kernel_alphas)
    with torch.no_grad():
        return float(mmd_loss(z_source, z_target).item())

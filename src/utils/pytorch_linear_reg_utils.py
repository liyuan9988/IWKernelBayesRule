import torch
from typing import Optional


def fit_linear(target: torch.Tensor,
               feature: torch.Tensor,
               reg: float = 0.0,
               sample_weight: Optional[torch.Tensor] = None):
    """
    Parameters
    ----------
    target: torch.Tensor[nBatch, dim1, dim2, ...]
    feature: torch.Tensor[nBatch, feature_dim]
    reg: float
        value of l2 regularizer
    sample_weight: torch.Tensor[nBatch, 1]
    Returns
    -------
        weight: torch.Tensor[feature_dim, dim1, dim2, ...]
            weight of ridge linear regression. weight.size()[0] = feature_dim+1 if add_intercept is true
    """
    assert feature.dim() == 2
    assert target.dim() >= 2
    nData, nDim = feature.size()
    if sample_weight is None:
        A = torch.matmul(feature.t(), feature)
    else:
        A = torch.matmul(feature.t(), sample_weight * feature)
    device = feature.device
    A = A + reg * torch.eye(nDim, device=device) * nData
    # U = torch.cholesky(A)
    # A_inv = torch.cholesky_inverse(U)
    # TODO use cholesky version in the latest pytorch
    A_inv = torch.inverse(A)
    if target.dim() == 2:
        if sample_weight is None:
            b = torch.matmul(feature.t(), target)
        else:
            b = torch.matmul(feature.t(), sample_weight * target)
        weight = torch.matmul(A_inv, b)
    else:
        if sample_weight is None:
            b = torch.einsum("nd,n...->d...", feature, target)
        else:
            b = torch.einsum("nd,n...->d...", feature, sample_weight * target)
        weight = torch.einsum("de,d...->e...", A_inv, b)
    return weight


def linear_reg_pred(feature: torch.Tensor, weight: torch.Tensor):
    assert weight.dim() >= 2
    if weight.dim() == 2:
        return torch.matmul(feature, weight)
    else:
        return torch.einsum("nd,d...->n...", feature, weight)


def linear_reg_loss(target: torch.Tensor,
                    feature: torch.Tensor,
                    reg: float,
                    sample_weight: Optional[torch.Tensor] = None):
    weight = fit_linear(target, feature, reg, sample_weight)
    pred = linear_reg_pred(feature, weight)
    nData, nDim = feature.size()
    error = (target - pred) ** 2
    if sample_weight is not None:
        error = sample_weight * error
    return torch.sum(error) + reg * torch.norm(weight) ** 2 * nData


def outer_prod(mat1: torch.Tensor, mat2: torch.Tensor):
    """
    Parameters
    ----------
    mat1: torch.Tensor[nBatch, mat1_dim1, mat1_dim2, mat1_dim3, ...]
    mat2: torch.Tensor[nBatch, mat2_dim1, mat2_dim2, mat2_dim3, ...]

    Returns
    -------
    res : torch.Tensor[nBatch, mat1_dim1, ..., mat2_dim1, ...]
    """

    mat1_shape = tuple(mat1.size())
    mat2_shape = tuple(mat2.size())
    assert mat1_shape[0] == mat2_shape[0]
    nData = mat1_shape[0]
    aug_mat1_shape = mat1_shape + (1,) * (len(mat2_shape) - 1)
    aug_mat1 = torch.reshape(mat1, aug_mat1_shape)
    aug_mat2_shape = (nData,) + (1,) * (len(mat1_shape) - 1) + mat2_shape[1:]
    aug_mat2 = torch.reshape(mat2, aug_mat2_shape)
    return aug_mat1 * aug_mat2


def add_const_col(mat: torch.Tensor):
    """

    Parameters
    ----------
    mat : torch.Tensor[n_data, n_col]

    Returns
    -------
    res : torch.Tensor[n_data, n_col+1]
        add one column only contains 1.

    """
    assert mat.dim() == 2
    n_data = mat.size()[0]
    device = mat.device
    return torch.cat([mat, torch.ones((n_data, 1), device=device)], dim=1)

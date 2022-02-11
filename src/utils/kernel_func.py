import numpy as np
from scipy.spatial.distance import cdist


class AbsKernel:
    def fit(self, data: np.ndarray, **kwargs) -> None:
        raise NotImplementedError

    def fit_from_dist_mat(self, dist_mat: np.ndarray, **kwargs) -> None:
        # Fit kernel matrix from Euclid distant matrix.
        raise NotImplementedError

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def cal_kernel_mat_grad(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        # Compute gradient of kernel matrix with respect to data2.
        raise NotImplementedError

    def cal_kernel_mat_from_dist_mat(self, dist_mat: np.ndarray) -> np.ndarray:
        # Compute kernel matrix from Euclid distant matrix.
        raise NotImplementedError


class LinearKernel(AbsKernel):

    def __init__(self):
        super(LinearKernel, self).__init__()

    def fit(self, data: np.ndarray, **kwargs) -> None:
        pass

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        return data1 @ data2.T


class BinaryKernel(AbsKernel):

    def __init__(self):
        super(BinaryKernel, self).__init__()

    def fit(self, data: np.ndarray, **kwargs):
        pass

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        res = data1.dot(data2.T)
        res += (1 - data1).dot(1 - data2.T)
        return res


class FourthOrderGaussianKernel(AbsKernel):
    bandwidth: np.float

    def fit(self, data: np.ndarray, **kwargs) -> None:
        dists = cdist(data, data, 'sqeuclidean')
        self.bandwidth = np.sqrt(np.median(dists))

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        diff_data = data1[:, np.newaxis, :] - data2[np.newaxis, :, :]
        u = diff_data / self.bandwidth
        kernel_tensor = np.exp(- u ** 2 / 2.0) * (3 - u ** 2) / 2.0 / np.sqrt(6.28)
        return np.product(kernel_tensor, axis=2)


class SixthOrderGaussianKernel(AbsKernel):
    bandwidth: np.float

    def fit(self, data: np.ndarray, **kwargs) -> None:
        dists = cdist(data, data, 'sqeuclidean')
        self.bandwidth = np.sqrt(np.median(dists))

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        diff_data = data1[:, np.newaxis, :] - data2[np.newaxis, :, :]
        u = diff_data / self.bandwidth
        kernel_tensor = np.exp(- u ** 2 / 2.0) * (15 - 10 * u ** 2 + u ** 4) / 8.0 / np.sqrt(6.28)
        return np.product(kernel_tensor, axis=2)


class FourthOrderEpanechnikovKernel(AbsKernel):
    bandwidth: np.float

    def fit(self, data: np.ndarray, **kwargs) -> None:
        n_data = data.shape[0]
        assert data.shape[1] == 1
        self.bandwidth = 3.03 * np.std(data) / (n_data ** 0.12)

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        assert data1.shape[1] == 1
        assert data2.shape[1] == 1
        dists = cdist(data1, data2, 'sqeuclidean') / (self.bandwidth ** 2)
        mat = (1.0 - dists) * (3 / 4) / self.bandwidth
        mat = np.maximum(mat, 0.0)
        mat = mat * (1.0 - 7 * dists / 3) * 15 / 8
        return mat


class EpanechnikovKernel(AbsKernel):
    bandwidth: np.float

    def fit(self, data: np.ndarray, **kwargs) -> None:
        n_data = data.shape[0]
        assert data.shape[1] == 1
        self.bandwidth = 2.34 * np.std(data) / (n_data ** 0.25)

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        assert data1.shape[1] == 1
        assert data2.shape[1] == 1
        dists = cdist(data1, data2, 'sqeuclidean') / (self.bandwidth ** 2)
        mat = (1.0 - dists) * (3 / 4) / self.bandwidth
        mat = np.maximum(mat, 0.0)
        return mat


class GaussianKernel(AbsKernel):
    sigma: np.float

    def __init__(self):
        super(GaussianKernel, self).__init__()

    def fit(self, data: np.ndarray, **kwargs) -> None:
        dists = cdist(data, data, 'sqeuclidean')
        self.sigma = np.median(dists)

    def fit_from_dist_mat(self, dist_mat: np.ndarray, **kwargs) -> None:
        self.sigma = np.median(dist_mat ** 2)

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        dists = cdist(data1, data2, 'sqeuclidean')
        dists = dists / self.sigma
        return np.exp(-dists)

    def cal_kernel_mat_grad(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        dists = cdist(data1, data2, 'sqeuclidean')
        dists = dists / self.sigma
        res = np.exp(-dists)[:, :, np.newaxis]
        res = res * (2 / self.sigma * (data1[:, np.newaxis, :] - data2[np.newaxis, :, :]))
        return res

    def cal_kernel_mat_from_dist_mat(self, dist_mat: np.ndarray):
        return np.exp(-(dist_mat ** 2) / self.sigma)


class IMQ(AbsKernel):
    sigma: float
    degree: float

    def __init__(self, degree: float = 0.5):
        super(IMQ, self).__init__()
        assert 0 < degree < 1
        self.degree = degree

    def fit(self, data: np.ndarray, **kwargs) -> None:
        dists = cdist(data, data, 'sqeuclidean')
        self.sigma = np.median(dists)

    def fit_from_dist_mat(self, dist_mat: np.ndarray, **kwargs) -> None:
        self.sigma = np.median(dist_mat ** 2)

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        dist = cdist(data1, data2, 'sqeuclidean')
        return np.power(dist/self.sigma + 1, -self.degree)

    def cal_kernel_mat_from_dist_mat(self, dist_mat: np.ndarray) -> np.ndarray:
        return np.power(dist_mat ** 2 / self.sigma + 1, -self.degree)

    def cal_kernel_mat_grad(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        dists = cdist(data1, data2, 'sqeuclidean')
        res = -self.degree * np.power(dists / self.sigma + 1, -self.degree-1)[:, :, np.newaxis]
        res = res * -(2 / self.sigma * (data1[:, np.newaxis, :] - data2[np.newaxis, :, :]))
        return res


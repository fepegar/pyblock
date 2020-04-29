import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from volume import Volume
from variogram import compute_variogram

BINS = 32


class Blockmatching:
    def __init__(self, ref_path, flo_path):
        self.ref = Volume(ref_path)
        self.flo = Volume(flo_path)


    def downsample(self):
        self.ref.downsample()
        self.flo.downsample()


    def get_joint_histogram(self, bins=BINS):
        return np.histogram2d(
            self.ref.current_data.ravel(),
            self.flo.current_data.ravel(),
            bins=bins,
        )


    def prob_xy(self, bins=BINS):
        counts, _, _ = self.get_joint_histogram(bins=bins)
        return counts / self.ref.current_data.size


    def mutual_information(self, bins=BINS, normalized=True):
        prob_xy = self.prob_xy(bins=bins)
        prob_x = prob_xy.sum(axis=0)
        prob_y = prob_xy.sum(axis=1)
        entropy_x = -np.sum(prob_x * np.log(prob_x))
        entropy_y = -np.sum(prob_y * np.log(prob_y))
        prob_xy_valid = prob_xy[np.nonzero(prob_xy)]
        entropy_joint = -np.sum(prob_xy_valid * np.log(prob_xy_valid))
        if normalized:
            return (entropy_x + entropy_y) / entropy_joint
        else:
            return entropy_x + entropy_y - entropy_joint


    def get_displacements(
            self, N=None, sigma=None, omega=None, delta=None):
        """
        N is the block size
        Ω is the half size of the search neighborhood in the reference when
            looking for similar blocks
        Δ = Δ1 is the block spacing in the floating image
        Σ = Δ2 is the step between blocks to be tested in the search neighborhood
        """
        si, sj = self.ref.current_data.shape
        if N is None:
            N = min(si, sj) // 8
        if omega is None:
            omega = N
        if delta is None:
            delta = N // 4
        if sigma is None:
            sigma = 4
        print(f'N = block size = {N}')
        print(f'Ω = search neighborhood = {omega}')
        print(f'Δ = Δ1 = block spacing flo = {delta}')
        print(f'Σ = Δ2 = block spacing ref = {sigma}')
        # N //= 2**3
        # omega //= 2**3
        # delta //= 2**3
        # sigma //= 2**3

        displacements = []
        stds = []
        for i in tqdm(range(0, si - N + 1, delta)):
            for j in range(0, sj - N + 1, delta):
                flo_block = self.flo.current_data[i:i + N, j:j + N]
                std = flo_block.std()
                if std < 5:  ### use percentile?
                    continue
                stds.append(std)
                flo_center = np.array((i + N / 2, j + N / 2)) - 0.5
                max_cc = -np.inf
                for k in range(i - omega, i + omega + 1, sigma):
                    for l in range(j - omega, j + omega + 1, sigma):
                        ref_block = self.ref.current_data[k:k + N, l:l + N]
                        if ref_block.shape != flo_block.shape:
                            continue
                        cc = self.cross_correlation(flo_block, ref_block)
                        if cc > max_cc:
                            ref_center = np.array((k + N / 2, l + N / 2)) - 0.5
                            max_cc = cc
                displacements.append((flo_center, ref_center, max_cc))
        flo_centers, ref_centers, ccs = tuple(zip(*displacements))
        flo_centers = np.array(flo_centers)
        ref_centers = np.array(ref_centers)
        return flo_centers, ref_centers, ccs, stds


    def cross_correlation(self, x, y, epsilon=1e-6):
        N = x.size
        numerator = (x - x.mean()) * (y - y.mean())
        denominator = x.std() * y.std()
        cc = 1 / N**2 * np.sum(numerator / denominator)
        return cc


    def plot_joint_histogram(self, bins=BINS, log=True):
        counts, bins_x, bins_y = self.get_joint_histogram(bins=bins)
        if log:
            counts += 1
            counts = np.log(counts)
        plt.imshow(counts)
        plt.show()


    def plot_displacements(self, points, displacements, color=None):
        X, Y = points.T
        U, V = displacements.T
        C = color
        plt.imshow(self.flo.current_data, cmap='gray')
        args = [X, Y, U, V]
        if color is not None:
            C = color
            args.append(C)
        plt.quiver(*args, units='xy')
        plt.show()


    def run(self):
        return


if __name__ == "__main__":
    b = Blockmatching('ref_slice.nii.gz', 'flo_slice.nii.gz')
    for _ in range(3):
        b.downsample()
    flo_centers, ref_centers, ccs, stds = b.get_displacements()
    flo_centers = flo_centers[::2]
    ref_centers = ref_centers[::2]
    # plt.hist(stds, 50)
    # plt.show()
    displacements = ref_centers - flo_centers
    # b.plot_displacements(flo_centers, displacements, color=ccs)
    h, dz = compute_variogram(flo_centers, displacements)

    x = []
    y = []
    for i in range(int(np.floor(h.max()))):  # 0 - 34
        bin_min = i + 0.5
        bin_max = bin_min + 1
        idx, = np.where((bin_min <= h) & (h < bin_max))
        bin_middle = np.mean((bin_min, bin_max))
        mean = dz[idx].mean()
        x.append(bin_middle)
        y.append(mean)

    # plt.scatter(h, dz + np.random.rand(dz.size).reshape(dz.shape) - 0.5, 3, alpha=0.25)
    plt.scatter(x, y)
    plt.show()

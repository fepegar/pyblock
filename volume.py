from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


class Volume:
    def __init__(self, path):
        self.path = Path(path)
        self.nifti = nib.load(str(self.path))
        self.data = self.nifti.get_fdata().squeeze()
        self.current_data = self.pad(self.data)
        self.current_downsample_factor = 1
        self.shape = self.data.shape


    def downsample(self):
        self.current_data = self.current_data[::2,::2]
        self.current_downsample_factor *= 2


    def pad(self, array):
        target = self.closest_powerof_two(max(array.shape), smaller=False)
        pad_width = target - np.array(array.shape)
        if not pad_width.any():  # already at max power of 2
            return array
        padded = np.pad(array, pad_width, mode='minimum')
        return padded


    def closest_powerof_two(self, n, smaller=True):
        """
        closest_powerof_two(513) = 512
        closest_powerof_two(512) = 256
        """
        p = np.log(n) / np.log(2)
        if p % 1 == 0:
            if smaller:
                p -= 1
            result = 2 ** p
        else:
            if smaller:
                result = 2 ** np.floor(p)
            else:
                result = 2 ** np.ceil(p)
        return int(result)


    def get_pyramid_shapes_map(self):
        shape = list(self.shape)

        level = 0
        shapes_map = {level: shape}

        last_level = False
        while not last_level:
            old_shape = shapes_map[level]
            max_dim = max(old_shape)
            closest_power = self.closest_powerof_two(max_dim)
            new_shape = [min(closest_power, n) for n in old_shape]
            level += 1
            shapes_map[level] = new_shape

            if max(new_shape) == 32:
                last_level = True

        return shapes_map


    def plot(self):
        plt.imshow(self.current_data)
        plt.show()


if __name__ == "__main__":
    path = 'ref_slice.nii.gz'
    volume = Volume(path)
    print(volume.get_pyramid_shapes_map())

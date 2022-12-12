import numpy as np
import flat_panel_project_utils as utils


class Grid:
    # initializes all non static class variables
    def __init__(self, width, height, spacing):
        self.width = width
        self.height = height
        self.spacing = spacing
        # from O = [-0.5(N0 - 1)s0, -0.5(N1 - 1)s1], s0 = spacing[0], s1 = spacing[1]
        self.origin = np.array([-0.5 * (self.width - 1.0) * self.spacing[0],
                                -0.5 * (self.height - 1.0) * self.spacing[1]])
        self.buffer = np.zeros([self.width, self.height])

    # allow external code to modify buffer
    def set_buffer(self, buffer):
        self.buffer = buffer

    # external code get the internal code
    def get_buffer(self):
        return self.buffer

    def get_origin(self):
        return self.origin

    def get_spacing(self):
        return self.spacing

    # size is the width * height
    def get_size(self):
        return np.array([self.width, self.height])

    # x = (e0, e1, o) * (p, 1)T,
    def index_to_physical(self, i, j):
        # page 7
        return np.array([self.spacing[0] * i + self.origin[0],
                         self.spacing[1] * j + self.origin[1]])
        # 2x3 dot 3x1 -----> 2x1

    # From world to image coordinates: P = [e0, e1] * [X - o] = [(x - Ox)/s0, (y - Oy)/s1]
    def physical_to_index(self, x, y):
        return np.array([(x - self.origin[0]) / self.spacing[0],
                         (y - self.origin[1]) / self.spacing[1]])

    # self.buffer = np.zero(self.height, self.width), (j, i)
    def set_at_index(self, i, j, index):
        self.buffer[i, j] = index

    def get_at_index(self, i, j):
        return self.buffer[i, j]

    # return i and j to utils.interpolate
    def get_at_physical(self, x, y):
        temp = self.physical_to_index(x, y)
        a = temp[0]
        b = temp[1]
        return utils.interpolate(self, a, b)


    #Exercise 2
    def set_origin(self, origin):
        self.origin = origin

    # Filtering

    def next_power_of_two(value):
        next_value = np.power(2, int(np.ceil(np.log2(value))) + 1)
        return next_value

        # 1. do fft to p(s,theta), got P(omega,theta)
        # 2. P(omega,theta) * H(omega, theta) = Q(omega,theta)
        # 3. do ifft to Q, got q(s,theta)
        # 4. Back Projection to q

    def ramp_filter(self, detector_spacing):
        # TODO: calculate the ramp kernel(pyramid like)
        sinogram = self
        filter = np.zeros((sinogram.height, sinogram.width))
        ramp = np.zeros(sinogram.width)
        sinogram_ramp = Grid(sinogram.width, sinogram.height, sinogram.spacing)
        sinogram_ramp.set_origin([0, (-0.5 * (sinogram.height - 1) * detector_spacing)])
        domain_o = int(sinogram.width / 2)  # Matrix, 1 * width

        for i in range(-domain_o, domain_o, 1):
            ramp[i] = np.abs(i / (sinogram.height * detector_spacing))

        for j in range(sinogram.height):
            temp_1 = np.fft.fft(sinogram.get_buffer()[j])
            result = np.multiply(temp_1, ramp)
            fft_ramp_1 = np.fft.ifft(result)
            fft_ramp_2 = np.real(fft_ramp_1)
            # if we use "np.abs", there would be more artificial parts at the reconstruction image
            filter[j] = fft_ramp_2  # Matrix, 1 * width

        sinogram_ramp.set_buffer(filter)
        return sinogram_ramp

    def ram_like_filter(self, detector_spacing):
        # TODO: Should transfer to RamLike filter from spatial to frequency domain
        sinogram = self
        # remember the fft of ramlike kernel
        filter = np.zeros((sinogram.height, sinogram.width))
        ram_like = np.zeros(sinogram.width)
        sinogramRamlike = Grid(sinogram.width, sinogram.height, sinogram.spacing)
        domain_o = int(sinogram.height / 2)

        for i in range(-domain_o, domain_o, 1):
            if i == 0:
                ram_like[i] = 1 / (4 * (detector_spacing ** 2))
            elif i % 2 == 1:
                # odd
                ram_like[i] = -1 / ((i * np.pi * detector_spacing) ** 2)
            else:
                # even
                ram_like[i] = 0
        for j in range(sinogram.height):
            temp = np.fft.fft(sinogram.get_buffer()[j])
            temp_fft = np.fft.fft(ram_like)
            res = np.multiply(temp, temp_fft)
            res_ifft = np.fft.ifft(res)
            res_real = np.real(res_ifft)
            filter[j] = res_real

        sinogramRamlike.set_buffer(filter)
        return sinogramRamlike
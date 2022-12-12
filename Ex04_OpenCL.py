import numpy as np
import pyopencl as cl

import flat_panel_project_utils as utils
from Grid import Grid


def create_sinogram(phantom, number_of_projections, detector_spacing, detector_sizeInPixels, angular_scan_range):
    base_angle = angular_scan_range / number_of_projections
    sino_grid = Grid(number_of_projections, detector_sizeInPixels, [base_angle, detector_spacing])
    '''
    set origin from grid: self.origin = 
    np.array([-0.5 * (self.width - 1) * self.spacing[0],
              -0.5 * (self.height - 1) * self.spacing[1]])'''
    sino_o_x = 0
    sino_o_y = -0.5 * (detector_sizeInPixels - 1) * detector_spacing
    sino_grid.set_origin([sino_o_x, sino_o_y])

    # we use the shape function from numpy.core.fromnumeric to calculate the phantom matrix
    phantom_grid = Grid(phantom.shape[1], phantom.shape[0], [0.5, 0.5])
    phantom_grid.set_buffer(phantom)
    # utils.show(phantom, "phantom")
    for N_of_P in range(number_of_projections):
        theta = base_angle * N_of_P

        # then use the detector_spacing and detector_sizeInPixels to express the x and y coordinates
        for d_size in range(0, detector_sizeInPixels):
            x = (d_size * detector_spacing + sino_o_y) * np.cos(theta * np.pi / 180)
            y = (d_size * detector_spacing + sino_o_y) * np.sin(theta * np.pi / 180)

            # the width and height of phantom size are equal
            # the length of the ray should be longer than 2 times the root of the width or height
            delta_t = 0.5
            sample_value = 0
            for L in np.arange(-phantom.shape[0], phantom.shape[0], delta_t):
                # the coordinates where the ray intersects the image is (x_physical, y_physical)
                x_physical = -np.sin(theta * np.pi / 180) * L + x
                y_physical = np.cos(theta * np.pi / 180) * L + y
                sample_value = sample_value + phantom_grid.get_at_physical(x_physical, y_physical) * delta_t

            sino_grid.set_at_index(N_of_P, d_size, sample_value)

    return sino_grid


def backprojectOpenCL(sinogram, size_x, size_y, spacing):

    number_of_projections = sinogram.get_size()[1]  # height
    detector_sizeInPixels = sinogram.get_size()[0]  # width
    detector_spacing = spacing[0]
    angular_increment = spacing[1]
    sino = np.array(sinogram.get_buffer()).astype(np.float32)

    ctx = cl.create_some_context()
    angle = 180 / number_of_projections

    queue = cl.CommandQueue(ctx)
    image = cl.image_from_array(ctx, sino, 1)
    a_np = np.zeros((size_y, size_x)).astype(np.float32)
    res_np = np.empty_like(a_np)

    mf = cl.mem_flags
    res_g = cl.Buffer(ctx, mf.READ_WRITE, a_np.nbytes)
    kernel = open("back.cl").read()
    prg = cl.Program(ctx, kernel).build()
    prg.sum(queue, a_np.shape, None, np.int32(number_of_projections), np.float32(angle), np.float32(detector_spacing),
            np.float32(angular_increment), np.int32(detector_sizeInPixels), image, res_g)

    cl.enqueue_copy(queue, res_np, res_g)

    return np.rot90(res_np, -1)


if __name__ == '__main__':
    phantom_size = 64
    sino = create_sinogram(phantom=utils.shepp_logan(phantom_size), number_of_projections=128,
                           detector_spacing=0.5, detector_sizeInPixels=128, angular_scan_range=180)
    utils.show(sino.get_buffer(), "Sino")
    # cv2.imsave(sino.get_buffer(), "Sinogram")
    backprojected = backprojectOpenCL(sino, phantom_size, phantom_size, [0.5, 0.5])
    utils.show(backprojected, "Backproject")


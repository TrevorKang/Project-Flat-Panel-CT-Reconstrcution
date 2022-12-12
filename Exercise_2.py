import numpy as np
import flat_panel_project_utils as utils
from Grid import Grid


def create_sinogram(phantom, number_of_projections, detector_spacing, detector_sizeInPixels, angular_scan_range):
    # TODO: to calculate p(s,θ)
    # base_angle is the base angle of rotation
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


# For each pixel in phantom, get the pixel value by iterating each angle of detector
# Sum up the values to itself


def backproject(sinogram, reco_size_x, reco_size_y, spacing):
    # TODO: reconstruction the phantom without filtering
    backproject = Grid(reco_size_x, reco_size_y, spacing)
    sinogram.set_origin([0, -((sinogram.height - 1) * 0.5) * sinogram.spacing[1]])
    for r_x in range(reco_size_x):
        for r_y in range(reco_size_y):
            v_theta_s = 0
            i2p = backproject.index_to_physical(r_x, r_y)
            # sinogram.width is number_of_projections
            for N_of_P in range(0, sinogram.width + 1):
                # sinogram.spacing[0] is the base angle
                theta = sinogram.spacing[0] * N_of_P
                ''' 
                sinogram.spacing[0] is detector_sizeInPixels, as well as the height of sinogram
                sinogram.spacing[1] is detector_spacing, as the width
                '''
                a = [i2p[0], i2p[1]]
                b = [np.cos(theta * np.pi / 180), np.sin(theta * np.pi / 180)]
                # project_s = i2p[0] *np.cos(theta * np.pi / 180) + i2p[1] * np.sin(theta * np.pi / 180)
                project_s = np.dot(a, b)
                # size = project / sinogram.spacing[1]
                v_theta_s += sinogram.get_at_physical(theta, project_s)
                backproject.set_at_index(r_x, r_y, v_theta_s)
    return backproject


if __name__ == '__main__':

    phantom_size = 128

    sinogram_parameter = create_sinogram(phantom=utils.shepp_logan(phantom_size), number_of_projections=128,
                                         detector_spacing=0.5,
                                         detector_sizeInPixels=128, angular_scan_range=180)
    # utils.show(sinogram_parameter.get_buffer(), "sinogram")

    back_project = backproject(sinogram_parameter, phantom_size, phantom_size, [0.5, 0.5])
    # utils.show(back_project, "Backproject")
    #
    ramp_test = sinogram_parameter.ramp_filter(sinogram_parameter.spacing[1])
    utils.show(ramp_test.get_buffer(), "ramp_test")
    FBP = backproject(ramp_test, phantom_size, phantom_size, [0.5, 0.5])
    # # utils.show(ramp_test.get_buffer(), "ramp_test")
    # utils.show(FBP, "FBP")
    #
    ramlike_test = sinogram_parameter.ram_like_filter(sinogram_parameter.spacing[1])
    # # utils.show(ramp_test.get_buffer(), "sino_test")
    FBP_ramlike = backproject(ramlike_test, phantom_size, phantom_size, [0.5, 0.5])
    utils.show(FBP_ramlike.get_buffer(), "FBP_ramlike")


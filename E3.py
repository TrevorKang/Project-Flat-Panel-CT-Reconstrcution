import numpy as np
import flat_panel_project_utils as utils
from Grid import Grid
from Exercise_2 import backproject


def create_fanogram(phantom, number_of_projections, detector_spacing, detector_sizeInPixels, angular_increment, d_si, d_sd):
    shepp_grid = Grid(phantom.shape[1], phantom.shape[0], [0.5, 0.5])
    shepp_grid.set_buffer(phantom)
    fano_grid = Grid(number_of_projections, detector_sizeInPixels, [angular_increment, detector_spacing])
    fano_o_x = 0
    fano_o_y = -0.5 * (detector_sizeInPixels - 1) * detector_spacing
    fano_grid.set_origin([fano_o_x, fano_o_y])
    delta = 0.5

    for N_of_P in range(number_of_projections):

        beta = N_of_P * angular_increment

        cos_beta = np.cos(beta * np.pi / 180)
        sin_beta = np.sin(beta * np.pi / 180)

        d_id = d_sd - d_si
        s = [-d_si * sin_beta, d_si * cos_beta]
        m = [d_id * sin_beta, -d_id * cos_beta]

        for t_index in range(0, detector_sizeInPixels):
            mp = [(fano_o_y + t_index * detector_spacing) * cos_beta, (fano_o_y + t_index * detector_spacing) * sin_beta]
            p = [m[0] + mp[0], m[1] + mp[1]]
            sp = [p[0] - s[0], p[1] - s[1]]
            d_sp = np.sqrt(np.power(sp[0], 2) + np.power(sp[1], 2))

            value_sample = 0
            # Sampling along SP
            for l in np.arange(0, d_sp, delta):
                x_physical = s[0] + l * (sp[0] / d_sp)
                y_physical = s[1] + l * (sp[1] / d_sp)

                value_sample = value_sample + shepp_grid.get_at_physical(x_physical, y_physical) * delta

            fano_grid.set_at_index(N_of_P, t_index, value_sample)
    utils.show(fano_grid.get_buffer(), "Fanoggram")
    return fano_grid


def rebinning(fanogram, d_si, d_sd):
    number_of_projections = 180
    detector_sizeInPixels = 180
    angular_increment = 1
    detector_spacing = 0.5
    sino_grid = Grid(number_of_projections, detector_sizeInPixels, [angular_increment, detector_spacing])
    sino_o_x = 0
    sino_o_y = -0.5 * (detector_sizeInPixels - 1) * detector_spacing
    sino_grid.set_origin([sino_o_x, sino_o_y])

    for s_index in range(detector_sizeInPixels):
        for N_of_P in range(0, number_of_projections):
            s = s_index * detector_spacing + sino_o_y
            gamma = np.arcsin(s / d_si) / (np.pi / 180)
            t = d_sd * np.tan(gamma * np.pi / 180)
            theta = N_of_P * angular_increment
            beta = theta - gamma
            value = 0
            # if (t < -detector_spacing * ((fanogram.height - 1) / 2)) or (
            #         t > detector_spacing * ((fanogram.height - 1) / 2)):
            #     sino_grid.buffer[:, s_index] = np.zeros(number_of_projections)
            # else:
            if beta < 0:
                beta = beta + 360
                gamma = - gamma
                beta = beta - 2 * gamma - 180
                t = -t
                value = fanogram.get_at_physical(beta, t)
            else:
                value = fanogram.get_at_physical(beta, t)

            sino_grid.set_at_index(N_of_P, s_index, value)

    utils.show(sino_grid.get_buffer(), "sino")

    return sino_grid


if __name__ == '__main__':
    phantom_size = 64
    number_of_projections = 240
    detector_spacing = 0.5
    detector_sizeInPixels = 160
    d_si = 95
    d_sd = 140

    fanogram_test = create_fanogram(phantom=utils.shepp_logan(phantom_size),
                                    number_of_projections=number_of_projections,
                                    detector_spacing=detector_spacing, detector_sizeInPixels=detector_sizeInPixels,
                                    angular_increment=1, d_si=d_si, d_sd=d_sd)
    rebinning_test = rebinning(fanogram=fanogram_test, d_si=d_si, d_sd=d_sd)

    ramp_sino = Grid.ramp_filter(rebinning_test, 0.5)
    FBP = backproject(ramp_sino, 64, 64, [1, 0.5])
    utils.show(FBP.get_buffer(), "FBP")

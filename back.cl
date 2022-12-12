__kernel void sum(int number_of_projections,float increment, float detector_spacing, float angular_scan_range, int detector_sizeInPixels,
__read_only image2d_t inputImage, __global float *res)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int width = get_global_size(1);
    int height = get_global_size(0);


    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
    for(int n = 0; n < number_of_projections; n++){
        float theta = n * increment / 180;
        float x_index = detector_spacing * x - 0.5 * (width - 1) * detector_spacing;
        float y_index = angular_scan_range * y - 0.5 * (height - 1) * angular_scan_range;
        float cos_theta = cos(theta * M_PI);
        float sin_theta = sin(theta * M_PI);

        float l = x_index * cos_theta + y_index * sin_theta;
        float origin = -(detector_sizeInPixels - 1) * detector_spacing / 2;
        float k = (l - origin) / detector_spacing;

        float2 location = (float2)(k+0.5f, n+0.5f);
        res[x + y * width] += read_imagef(inputImage, sampler, location).x;
    }
}

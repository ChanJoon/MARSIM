// camera.fs — pinhole fragment shader.
// Discards fragments beyond [near, far] and encodes metric depth (metres)
// in the red channel.  opengl_sim.hpp read_depth() reads GL_RED GL_FLOAT
// to get a metric depth buffer without NDC linearisation.
#version 330 core
out vec4 FragColor;

in vec3 ourColor;
in float vMetricDepth;   // metric view-space depth from camera.vs

uniform vec2 range;      // range.x = near_clip, range.y = far_clip

void main()
{
    if (vMetricDepth < range.x || vMetricDepth > range.y)
    {
        discard;
    }
    // Red channel = metric depth in metres.  GB channels unused by readback
    // but set to 0 so rqt_image_view displays a greyscale-ish image.
    FragColor = vec4(vMetricDepth, 0.0f, 0.0f, 1.0f);
}

// camera.vs — pinhole vertex shader for opengl_sim.hpp depth_pinhole mode.
// Branch 3 fix (option b): clip-space gl_Position.z range check is REMOVED.
// Instead, metric view-space depth is passed as a varying to the fragment
// shader, which discards fragments beyond [near, far] and outputs depth in
// the red channel as a metric value (meters).  The read_depth() pinhole path
// in opengl_sim.hpp reads GL_RED (metric meters) rather than linearising the
// GL depth buffer.
//
// Choice rationale: option (b) is simpler than option (a) (glDepthRange
// setup) because the GL depth buffer is not used for depth readback in
// pinhole mode — we read the red channel.  Discarding in the fragment shader
// is a one-liner that integrates cleanly with the existing camera.fs
// FragColor approach.
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

out vec3 ourColor;
out float vMetricDepth;   // view-space depth in metres, passed to FS

uniform mat4 view;
uniform mat4 projection;
uniform vec2 range;       // range.x = near_clip, range.y = far_clip

void main()
{
    vec4 viewPos = view * vec4(aPos, 1.0f);
    // OpenGL camera looks down -Z in view space; metric depth is positive.
    float metric_depth = -viewPos.z;
    vMetricDepth = metric_depth;

    gl_Position = projection * viewPos;
    ourColor = aColor;
}

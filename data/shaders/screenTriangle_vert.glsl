#version 450

layout (location = 0) out vec2 outUV;

vec2 positions[3] = vec2[](
vec2( 1.0, -3.0),
vec2( 1.0,  1.0),
vec2(-3.0,  1.0)
);

void main()
{
    outUV = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);

    // rotate uvs 90 anticlockwise
    outUV = vec2(outUV.y, 1.0 - outUV.x);

    gl_Position = vec4(outUV * 2.0f - 1.0f, 0.0f, 1.0f);
}

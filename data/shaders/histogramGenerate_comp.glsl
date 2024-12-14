#version 450
#extension GL_GOOGLE_include_directive : require

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout (binding = 5) uniform sampler2D texSampler;
layout (std430, binding = 8) buffer histogram { uint bins[256]; };
layout (binding = 2) uniform PostProcessBufferObject
{
    vec2 texDimensions;

    float minimumLuminance;
    float maximumLuminance;
    float timeDelta;
    float lumAdaptTau;

// only when using Reinhard
    float whitepoint;

// only when using Uchimura
    float maxDisplayBrigtness;
    float contrast;
    float linearSectionStart;
    float linearSectionLength;
    float black;
    float pedestal;

// only when using Lottes
    float a;
    float d;
    float hdrMax;
    float midIn;
    float midOut;

    int tonemapCurve;
} postProcessParameters;

#define EPSILON 0.005
float log2minLum = log2(postProcessParameters.minimumLuminance);
float invLog2lumRange = 1.0/log2(postProcessParameters.maximumLuminance);
shared uint HistogramShared[256];

float getLuminance(vec3 color)
{
    return dot(color, vec3(0.2127, 0.7152, 0.0722));
}

uint HDRToHistogramBin(vec3 hdrColor)
{
    float luminance = getLuminance(hdrColor);

    if (luminance < EPSILON)
    {
        return 0;
    }

    float logLum = clamp((log2(luminance) - log2minLum) * invLog2lumRange, 0.0, 1.0);
    return uint(logLum * 254.0 + 1.0);
}

void main()
{
    if(gl_GlobalInvocationID.x > postProcessParameters.texDimensions.x ||
    gl_GlobalInvocationID.y > postProcessParameters.texDimensions.y)
    {
        return;
    }
    vec2 uv = vec2(gl_GlobalInvocationID.xy) / postProcessParameters.texDimensions;

    HistogramShared[gl_LocalInvocationIndex] = 0;
    groupMemoryBarrier();
    barrier();

    /* Multiply by sun luminance */
    vec3 hdrCol = texture(texSampler, uv).rgb * 120000.0;
    uint binIndex = HDRToHistogramBin(hdrCol);
    atomicAdd(HistogramShared[binIndex], 1);

    groupMemoryBarrier();
    barrier();

    atomicAdd(bins[gl_LocalInvocationIndex], HistogramShared[gl_LocalInvocationIndex]);
}

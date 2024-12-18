#version 450
#extension GL_GOOGLE_include_directive : require

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout (std430, binding = 9) buffer averageLuminance { float avgLum; };
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
float log2lumRange = log2(postProcessParameters.maximumLuminance);
shared uint HistogramShared[256];

void main()
{
    uint countForThisBin = bins[gl_LocalInvocationIndex];
    HistogramShared[gl_LocalInvocationIndex] = countForThisBin * gl_LocalInvocationIndex;

    groupMemoryBarrier();
    barrier();

    bins[gl_LocalInvocationIndex] = 0;

    if(gl_LocalInvocationID.z < 128)
    {
        HistogramShared[gl_LocalInvocationIndex] += HistogramShared[gl_LocalInvocationIndex + 128];
    }
    groupMemoryBarrier();
    barrier();
    if(gl_LocalInvocationID.z < 64)
    {
        HistogramShared[gl_LocalInvocationIndex] += HistogramShared[gl_LocalInvocationIndex + 64];
    }
    groupMemoryBarrier();
    barrier();
    if(gl_LocalInvocationID.z < 32)
    {
        HistogramShared[gl_LocalInvocationIndex] += HistogramShared[gl_LocalInvocationIndex + 32];
    }
    groupMemoryBarrier();
    barrier();
    if(gl_LocalInvocationID.z < 16)
    {
        HistogramShared[gl_LocalInvocationIndex] += HistogramShared[gl_LocalInvocationIndex + 16];
    }
    groupMemoryBarrier();
    barrier();
    if(gl_LocalInvocationID.z < 8)
    {
        HistogramShared[gl_LocalInvocationIndex] += HistogramShared[gl_LocalInvocationIndex + 8];
    }
    groupMemoryBarrier();
    barrier();
    if(gl_LocalInvocationID.z < 4)
    {
        HistogramShared[gl_LocalInvocationIndex] += HistogramShared[gl_LocalInvocationIndex + 4];
    }
    groupMemoryBarrier();
    barrier();
    if(gl_LocalInvocationID.z < 2)
    {
        HistogramShared[gl_LocalInvocationIndex] += HistogramShared[gl_LocalInvocationIndex + 2];
    }
    groupMemoryBarrier();
    barrier();
    if(gl_LocalInvocationID.z < 1)
    {
        HistogramShared[gl_LocalInvocationIndex] += HistogramShared[gl_LocalInvocationIndex + 1];
    }
    groupMemoryBarrier();
    barrier();

    if(gl_GlobalInvocationID.x == 0)
    {
        float weightedLogAverage =
        (HistogramShared[0] / max(postProcessParameters.texDimensions.x * postProcessParameters.texDimensions.y - float(countForThisBin), 1.0)) - 1.0;

        float weightedAvgLum = exp2(((weightedLogAverage / 254.0) * log2lumRange) + log2minLum);
        float lumLastFrame = avgLum;
        float adaptedLum = lumLastFrame + (weightedAvgLum - lumLastFrame) *
        (1.0 - exp(-postProcessParameters.timeDelta * postProcessParameters.lumAdaptTau));
        if(lumLastFrame != 0)
        {
            avgLum = adaptedLum;
        } else {
            avgLum = weightedAvgLum;
        }
    }
}

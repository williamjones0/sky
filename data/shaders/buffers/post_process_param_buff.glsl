layout (set = 2, binding = 0) uniform PostProcessBufferObject
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
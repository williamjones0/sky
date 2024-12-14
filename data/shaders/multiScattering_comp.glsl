#version 450

layout (local_size_x = 1, local_size_y = 1, local_size_z = 64) in;

#extension GL_GOOGLE_include_directive : require
const float PI = 3.1415926535897932384626433832795;
const float PLANET_RADIUS_OFFSET = 0.01;

/* Return sqrt clamped to 0 */
float safeSqrt(float x)
{
    return sqrt(max(0, x));
}

float fromSubUvsToUnit(float u, float resolution) {
    return (u - 0.5 / resolution) * (resolution / (resolution - 1.0));
}

float fromUnitToSubUvs(float u, float resolution) {
    return (u + 0.5f / resolution) * (resolution / (resolution + 1.0));
}



/**
 * Get parameters used for skyViewLUT computation for texel with provided uv coords
 * @param uv - texel uv in the range [0,1]
 * @param atmosphereBoundaries - x is atmosphere bottom radius y is top
 * @param skyViewDimensions - skyViewLUT dimensions
 * @param viewHeight - viewHeight in world coordinates -> distance from planet center
 * @return - viewZenithAngle in x, lightViewAngle in y
 */
vec2 UvToSkyViewLUTParams(vec2 uv, vec2 atmosphereBoundaries, vec2 skyViewDimensions,
float viewHeight)
{
    /* Constrain uvs to valid sub texel range
    (avoid zenith derivative issue making LUT usage visible) */
    uv = vec2(fromSubUvsToUnit(uv.x, skyViewDimensions.x),
    fromSubUvsToUnit(uv.y, skyViewDimensions.y));

    float beta = asin(atmosphereBoundaries.x / viewHeight);
    float zenithHorizonAngle = PI - beta;

    float viewZenithAngle;
    float lightViewAngle;
    /* Nonuniform mapping near the horizon to avoid artefacts */
    if(uv.y < 0.5)
    {
        float coord = 1.0 - (1.0 - 2.0 * uv.y) * (1.0 - 2.0 * uv.y);
        viewZenithAngle = zenithHorizonAngle * coord;
    } else {
        float coord = (uv.y * 2.0 - 1.0) * (uv.y * 2.0 - 1.0);
        viewZenithAngle = zenithHorizonAngle + beta * coord;
    }
    lightViewAngle = (uv.x * uv.x) * PI;
    return vec2(viewZenithAngle, lightViewAngle);
}

/**
 * Get parameters used for skyViewLUT computation for texel with provided uv coords
 * @param intersectGround - true if ray intersects ground false otherwise
 * @param LUTParams - viewZenithAngle in x, lightViewAngle in y
 * @param viewHeight - viewHeight in world coordinates -> distance from planet center
 * @param atmosphereBoundaries - x is atmosphere bottom radius y is top
 * @param skyViewDimensions - skyViewLUT dimensions
 * @return - uv for the skyViewLUT sampling
 */
vec2 SkyViewLutParamsToUv(bool intersectGround, vec2 LUTParams, float viewHeight,
vec2 atmosphereBoundaries, vec2 skyViewDimensions)
{
    vec2 uv;
    float beta = asin(atmosphereBoundaries.x / viewHeight);
    float zenithHorizonAngle = PI - beta;

    if(!intersectGround)
    {
        float coord = LUTParams.x / zenithHorizonAngle;
        coord = (1.0 - safeSqrt(1.0 - coord)) / 2.0;
        uv.y = coord;
    } else {
        float coord = (LUTParams.x - zenithHorizonAngle) / beta;
        coord = (safeSqrt(coord) + 1.0) / 2.0;
        uv.y = coord;
    }
    uv.x = safeSqrt(LUTParams.y / PI);
    uv = vec2(fromUnitToSubUvs(uv.x, skyViewDimensions.x),
    fromUnitToSubUvs(uv.y, skyViewDimensions.y));
    return uv;
}

/**
 * Transmittance LUT uses not uniform mapping -> transfer from uv to this mapping
 * @param uv - uv in the range [0,1]
 * @param atmosphereBoundaries - x is atmoBottom radius, y is top radius
 * @return - height in x, zenith cos angle in y
 */
vec2 UvToTransmittanceLUTParams(vec2 uv, vec2 atmosphereBoundaries)
{
    /* Params.x stores height, Params.y stores ZenithCosAngle */
    vec2 Params;
    float H = safeSqrt(
    atmosphereBoundaries.y * atmosphereBoundaries.y
    - atmosphereBoundaries.x * atmosphereBoundaries.x);

    float rho = H * uv.y;
    Params.x = safeSqrt( rho * rho + atmosphereBoundaries.x * atmosphereBoundaries.x);

    float d_min = atmosphereBoundaries.y - Params.x;
    float d_max = rho + H;
    float d = d_min + uv.x * (d_max - d_min);

    Params.y = d == 0.0 ? 1.0 : (H * H - rho * rho - d * d) / (2.0 * Params.x * d);
    Params.y = clamp(Params.y, -1.0, 1.0);

    return Params;
}

/**
 * Transmittance LUT uses not uniform mapping -> transfer from mapping to texture uv
 * @param parameters - height in x, zenith cos angle in y
 * @param atmosphereBoundaries - x is bottom radius, y is top radius
 * @return - uv of the corresponding texel
 */
vec2 TransmittanceLUTParamsToUv(vec2 parameters, vec2 atmosphereBoundaries)
{
    float H = safeSqrt(
    atmosphereBoundaries.y * atmosphereBoundaries.y
    - atmosphereBoundaries.x * atmosphereBoundaries.x);

    float rho = safeSqrt(parameters.x * parameters.x -
    atmosphereBoundaries.x * atmosphereBoundaries.x);

    float discriminant = parameters.x * parameters.x * (parameters.y * parameters.y - 1.0) +
    atmosphereBoundaries.y * atmosphereBoundaries.y;
    /* Distance to top atmosphere boundary */
    float d = max(0.0, (-parameters.x * parameters.y + safeSqrt(discriminant)));

    float d_min = atmosphereBoundaries.y - parameters.x;
    float d_max = rho + H;
    float mu = (d - d_min) / (d_max - d_min);
    float r = rho / H;

    return vec2(mu, r);
}

/**
 * Return distance of the first intersection between ray and sphere
 * @param r0 - ray origin
 * @param rd - normalized ray direction
 * @param s0 - sphere center
 * @param sR - sphere radius
 * @return return distance of intersection or -1.0 if there is no intersection
 */
float raySphereIntersectNearest(vec3 r0, vec3 rd, vec3 s0, float sR)
{
    float a = dot(rd, rd);
    vec3 s0_r0 = r0 - s0;
    float b = 2.0 * dot(rd, s0_r0);
    float c = dot(s0_r0, s0_r0) - (sR * sR);
    float delta = b * b - 4.0*a*c;
    if (delta < 0.0 || a == 0.0)
    {
        return -1.0;
    }
    float sol0 = (-b - safeSqrt(delta)) / (2.0*a);
    float sol1 = (-b + safeSqrt(delta)) / (2.0*a);
    if (sol0 < 0.0 && sol1 < 0.0)
    {
        return -1.0;
    }
    if (sol0 < 0.0)
    {
        return max(0.0, sol1);
    }
    else if (sol1 < 0.0)
    {
        return max(0.0, sol0);
    }
    return max(0.0, min(sol0, sol1));
}

/**
 * Moves to the nearest intersection with top of the atmosphere in the direction specified in
 * worldDirection
 * @param worldPosition - current world position -> will be changed to new pos at the top of
 * 		the atmosphere if there exists such intersection
 * @param worldDirecion - the direction in which the shift will be done
 * @param atmosphereBoundaries - x is bottom radius, y is top radius
 */
bool moveToTopAtmosphere(inout vec3 worldPosition, vec3 worldDirection, vec2 atmosphereBoundaries)
{
    vec3 planetOrigin = vec3(0.0, 0.0, 0.0);
    /* Check if the worldPosition is outside of the atmosphere */
    if(length(worldPosition) > atmosphereBoundaries.y)
    {
        float distToTopAtmosphereIntersection = raySphereIntersectNearest(
        worldPosition, worldDirection, planetOrigin, atmosphereBoundaries.y);

        /* No intersection with the atmosphere */
        if (distToTopAtmosphereIntersection == -1.0) { return false; }
        else
        {
            vec3 upOffset = normalize(worldPosition) * -PLANET_RADIUS_OFFSET;
            worldPosition += worldDirection * distToTopAtmosphereIntersection + upOffset;
        }
    }
    /* Position is in or at the top of the atmosphere */
    return true;
}


layout(binding = 0) uniform CommonParamBufferObject
{
    mat4 model;
    mat4 view;
    mat4 proj;
    mat4 lHviewProj;
    float time;
} commonParameters;

layout(binding = 1) uniform AtmosphereParametersBuffer
{
    vec3 solar_irradiance;
    float sun_angular_radius;

    vec4 absorption_extinction;

    vec3 rayleigh_scattering;
    float mie_phase_function_g;

    vec3 mie_scattering;
    float bottom_radius;

    vec3 mie_extinction;
    float top_radius;

    vec4 mie_absorption;
    vec4 ground_albedo;

    vec4 rayleigh_density[3];
    vec4 mie_density[3];
    vec4 absorption_density[3];

    vec2 TransmittanceTexDimensions;
    vec2 MultiscatteringTexDimensions;
    vec2 SkyViewTexDimensions;
    vec2 _pad0;

    vec4 AEPerspectiveTexDimensions;

    vec4 sun_direction;
    vec4 camera_position;
} atmosphereParameters;

layout (binding = 3, rgba16f) uniform readonly image2D transmittanceLUT;
layout (binding = 4, rgba16f) uniform  image2D multiscatteringLUT;
/* ================================== NOT USED ==================================== */
layout (binding = 5, rgba16f) uniform readonly image2D skyViewLUT;
layout (binding = 6, rgba16f) uniform readonly image3D AEPerspective;
/* ================================================================================ */

/* This number should match the number of local threads -> z dimension */
const float SPHERE_SAMPLES = 64.0;
const float GOLDEN_RATIO = 1.6180339;
const float uniformPhase = 1.0 / (4.0 * PI);

shared vec3 MultiscattSharedMem[64];
shared vec3 LSharedMem[64];

struct RaymarchResult
{
    vec3 Luminance;
    vec3 Multiscattering;
};


/* ============================= MEDIUM SAMPLING ============================ */
vec3 SampleMediumExtinction(vec3 worldPosition)
{
    const float viewHeight = length(worldPosition) - atmosphereParameters.bottom_radius;

    const float densityMie = exp(atmosphereParameters.mie_density[1].w * viewHeight);
    const float densityRay = exp(atmosphereParameters.rayleigh_density[1].w * viewHeight);
    const float densityOzo = clamp(viewHeight < atmosphereParameters.absorption_density[0].x ?
    atmosphereParameters.absorption_density[0].w * viewHeight + atmosphereParameters.absorption_density[1].x :
    atmosphereParameters.absorption_density[2].x * viewHeight + atmosphereParameters.absorption_density[2].y,
    0.0, 1.0);

    vec3 mieExtinction = atmosphereParameters.mie_extinction * densityMie;
    vec3 rayleighExtinction = atmosphereParameters.rayleigh_scattering * densityRay;
    vec3 ozoneExtinction = atmosphereParameters.absorption_extinction.rgb * densityOzo;

    return mieExtinction + rayleighExtinction + ozoneExtinction;
}

vec3 SampleMediumScattering(vec3 worldPosition)
{
    const float viewHeight = length(worldPosition) - atmosphereParameters.bottom_radius;

    const float densityMie = exp(atmosphereParameters.mie_density[1].w * viewHeight);
    const float densityRay = exp(atmosphereParameters.rayleigh_density[1].w * viewHeight);
    const float densityOzo = clamp(viewHeight < atmosphereParameters.absorption_density[0].x ?
    atmosphereParameters.absorption_density[0].w * viewHeight + atmosphereParameters.absorption_density[1].x :
    atmosphereParameters.absorption_density[2].x * viewHeight + atmosphereParameters.absorption_density[2].y,
    0.0, 1.0);

    vec3 mieScattering = atmosphereParameters.mie_scattering * densityMie;
    vec3 rayleighScattering = atmosphereParameters.rayleigh_scattering * densityRay;
    /* Not considering ozon scattering in current version of this model */
    vec3 ozoneScattering = vec3(0.0, 0.0, 0.0);

    return mieScattering + rayleighScattering + ozoneScattering;
}
/* ========================================================================== */

RaymarchResult IntegrateScatteredLuminance(vec3 worldPosition, vec3 worldDirection,
vec3 sunDirection, float sampleCount)
{
    RaymarchResult result = RaymarchResult(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0));
    vec3 planet0 = vec3(0.0, 0.0, 0.0);
    float planetIntersectionDistance = raySphereIntersectNearest(
    worldPosition, worldDirection, planet0, atmosphereParameters.bottom_radius);
    float atmosphereIntersectionDistance = raySphereIntersectNearest(
    worldPosition, worldDirection, planet0, atmosphereParameters.top_radius);

    float integrationLength;
    /* ============================= CALCULATE INTERSECTIONS ============================ */
    if((planetIntersectionDistance == -1.0) && (atmosphereIntersectionDistance == -1.0)){
        /* ray does not intersect planet or atmosphere -> no point in raymarching*/
        return result;
    }
    else if((planetIntersectionDistance == -1.0) && (atmosphereIntersectionDistance > 0.0)){
        /* ray intersects only atmosphere */
        integrationLength = atmosphereIntersectionDistance;
    }
    else if((planetIntersectionDistance > 0.0) && (atmosphereIntersectionDistance == -1.0)){
        /* ray intersects only planet */
        integrationLength = planetIntersectionDistance;
    } else {
        /* ray intersects both planet and atmosphere -> return the first intersection */
        integrationLength = min(planetIntersectionDistance, atmosphereIntersectionDistance);
    }
    float integrationStep = integrationLength / float(sampleCount);

    vec2 atmosphereBoundaries = vec2(atmosphereParameters.bottom_radius, atmosphereParameters.top_radius);

    /* stores accumulated transmittance during the raymarch process */
    vec3 accumTrans = vec3(1.0, 1.0, 1.0);
    /* stores accumulated light contribution during the raymarch process */
    vec3 accumLight = vec3(0.0, 0.0, 0.0);
    float oldRayShift = 0;

    /* ============================= RAYMARCH ==========================================  */
    for(int i = 0; i < sampleCount; i++)
    {
        /* Sampling at 1/3rd of the integration step gives better results for exponential
           functions */
        float newRayShift = integrationLength * (float(i) + 0.3) / sampleCount;
        integrationStep = newRayShift - oldRayShift;
        vec3 newPos = worldPosition + newRayShift * worldDirection;
        oldRayShift = newRayShift;

        /* Raymarch shifts the angle to the sun a bit recalculate */
        vec3 upVector = normalize(newPos);
        vec2 transLUTParams = vec2( length(newPos),dot(sunDirection, upVector));

        /* uv coordinates later used to sample transmittance texture */
        vec2 transUV = TransmittanceLUTParamsToUv(transLUTParams, atmosphereBoundaries);
        /* because here transmittanceLUT is image and not a texture transfer
           from [0,1] -> [tex_width, tex_height] */
        ivec2 transImageCoords = ivec2(transUV * atmosphereParameters.TransmittanceTexDimensions);

        vec3 transmittanceToSun = vec3(imageLoad(transmittanceLUT, transImageCoords).rgb);
        vec3 mediumScattering = SampleMediumScattering(newPos);
        vec3 mediumExtinction = SampleMediumExtinction(newPos);

        /* TODO: This probably should be a texture lookup*/
        vec3 transIncreseOverInegrationStep = exp(-(mediumExtinction * integrationStep));
        /* Check if current position is in earth's shadow */
        float earthIntersectionDistance = raySphereIntersectNearest(
        newPos, sunDirection, planet0 + PLANET_RADIUS_OFFSET * upVector, atmosphereParameters.bottom_radius);
        float inEarthShadow = earthIntersectionDistance == -1.0 ? 1.0 : 0.0;

        /* Light arriving from the sun to this point */
        vec3 sunLight = inEarthShadow * transmittanceToSun * mediumScattering * uniformPhase;
        vec3 multiscatteredContInt =
        (mediumScattering - mediumScattering * transIncreseOverInegrationStep) / mediumExtinction;
        vec3 inscatteredContInt =
        (sunLight - sunLight * transIncreseOverInegrationStep) / mediumExtinction;
        /* For some reson I need to do this to avoid nans in multiscatteredLightInt ->
           precision error? */
        if(all(equal(transIncreseOverInegrationStep, vec3(1.0)))) {
            multiscatteredContInt = vec3(0.0);
            inscatteredContInt = vec3(0.0);
        }
        result.Multiscattering += accumTrans * multiscatteredContInt;
        accumLight += accumTrans * inscatteredContInt;
        // accumLight = accumTrans;
        accumTrans *= transIncreseOverInegrationStep;
    }
    result.Luminance = accumLight;
    return result;
    /* TODO: Check for bounced light off the earth */
}

void main()
{
    const float sampleCount = 20;

    vec2 uv = (vec2(gl_GlobalInvocationID.xy) + vec2(0.5, 0.5)) / atmosphereParameters.MultiscatteringTexDimensions;
    uv = vec2(fromSubUvsToUnit(uv.x, atmosphereParameters.MultiscatteringTexDimensions.x),
    fromSubUvsToUnit(uv.y, atmosphereParameters.MultiscatteringTexDimensions.y));

    /* Mapping uv to multiscattering LUT parameters
       TODO -> Is the range from 0.0 to -1.0 really needed? */
    float sunCosZenithAngle = uv.x * 2.0 - 1.0;
    vec3 sunDirection = vec3(
    0.0,
    sqrt(clamp(1.0 - sunCosZenithAngle * sunCosZenithAngle, 0.0, 1.0)),
    sunCosZenithAngle
    );

    float viewHeight = atmosphereParameters.bottom_radius +
    clamp(uv.y + PLANET_RADIUS_OFFSET, 0.0, 1.0) *
    (atmosphereParameters.top_radius - atmosphereParameters.bottom_radius - PLANET_RADIUS_OFFSET);

    vec3 worldPosition = vec3(0.0, 0.0, viewHeight);

    float sampleIdx = gl_LocalInvocationID.z;
    // local thread dependent raymarch
    {
        #define USE_HILL_SAMPLING 0
        #if USE_HILL_SAMPLING
        #define SQRTSAMPLECOUNT 8
        const float sqrtSample = float(SQRTSAMPLECOUNT);
        float i = 0.5 + float(sampleIdx / SQRTSAMPLECOUNT);
        float j = 0.5 + mod(sampleIdx, SQRTSAMPLECOUNT);
        float randA = i / sqrtSample;
        float randB = j / sqrtSample;

        float theta = 2.0 * PI * randA;
        float phi = PI * randB;
        #else
        /* Fibbonaci lattice -> http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/ */
        float theta = acos( 1.0 - 2.0 * (sampleIdx + 0.5) / SPHERE_SAMPLES );
        float phi = (2 * PI * sampleIdx) / GOLDEN_RATIO;
        #endif


        vec3 worldDirection = vec3( cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi));
        RaymarchResult result = IntegrateScatteredLuminance(worldPosition, worldDirection,
        sunDirection, sampleCount);

        MultiscattSharedMem[gl_LocalInvocationID.z] = result.Multiscattering / SPHERE_SAMPLES;
        LSharedMem[gl_LocalInvocationID.z] = result.Luminance / SPHERE_SAMPLES;
    }

    groupMemoryBarrier();
    barrier();

    if(gl_LocalInvocationID.z < 32)
    {
        MultiscattSharedMem[gl_LocalInvocationID.z] += MultiscattSharedMem[gl_LocalInvocationID.z + 32];
        LSharedMem[gl_LocalInvocationID.z] += LSharedMem[gl_LocalInvocationID.z + 32];
    }
    groupMemoryBarrier();
    barrier();
    if(gl_LocalInvocationID.z < 16)
    {
        MultiscattSharedMem[gl_LocalInvocationID.z] += MultiscattSharedMem[gl_LocalInvocationID.z + 16];
        LSharedMem[gl_LocalInvocationID.z] += LSharedMem[gl_LocalInvocationID.z + 16];
    }
    groupMemoryBarrier();
    barrier();
    if(gl_LocalInvocationID.z < 8)
    {
        MultiscattSharedMem[gl_LocalInvocationID.z] += MultiscattSharedMem[gl_LocalInvocationID.z + 8];
        LSharedMem[gl_LocalInvocationID.z] += LSharedMem[gl_LocalInvocationID.z + 8];
    }
    groupMemoryBarrier();
    barrier();
    if(gl_LocalInvocationID.z < 4)
    {
        MultiscattSharedMem[gl_LocalInvocationID.z] += MultiscattSharedMem[gl_LocalInvocationID.z + 4];
        LSharedMem[gl_LocalInvocationID.z] += LSharedMem[gl_LocalInvocationID.z + 4];
    }
    groupMemoryBarrier();
    barrier();
    if(gl_LocalInvocationID.z < 2)
    {
        MultiscattSharedMem[gl_LocalInvocationID.z] += MultiscattSharedMem[gl_LocalInvocationID.z + 2];
        LSharedMem[gl_LocalInvocationID.z] += LSharedMem[gl_LocalInvocationID.z + 2];
    }
    groupMemoryBarrier();
    barrier();
    if(gl_LocalInvocationID.z < 1)
    {
        MultiscattSharedMem[gl_LocalInvocationID.z] += MultiscattSharedMem[gl_LocalInvocationID.z + 1];
        LSharedMem[gl_LocalInvocationID.z] += LSharedMem[gl_LocalInvocationID.z + 1];
    }
    groupMemoryBarrier();
    barrier();
    if(gl_LocalInvocationID.z != 0)
    return;

    vec3 MultiscattSum = MultiscattSharedMem[0];
    vec3 InScattLumSum = LSharedMem[0];

    const vec3 r = MultiscattSum;
    const vec3 SumOfAllMultiScatteringEventsContribution = vec3(1.0/ (1.0 -r.x),1.0/ (1.0 -r.y),1.0/ (1.0 -r.z));
    vec3 Lum = InScattLumSum * SumOfAllMultiScatteringEventsContribution;

    imageStore(multiscatteringLUT, ivec2(gl_GlobalInvocationID.xy), vec4(Lum, 1.0));

}

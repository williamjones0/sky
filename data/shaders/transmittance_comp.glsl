#version 450
layout (local_size_x = 8, local_size_y = 4) in;

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

layout (binding = 3, rgba16f) uniform image2D transmittanceLUT;
/* ================================== NOT USED ==================================== */
layout (binding = 4, rgba16f) uniform readonly image2D multiscatteringLUT;
layout (binding = 5, rgba16f) uniform readonly image2D skyViewLUT;
layout (binding = 6, rgba16f) uniform readonly image3D AEPerspective;
/* ================================================================================ */

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

vec3 IntegrateTransmittance(vec3 worldPosition, vec3 worldDirection, uint sampleCount)
{
    vec3 planet0 = vec3(0.0, 0.0, 0.0);
    /* The length of ray between position and nearest atmosphere top boundary */
    float integrationLength = raySphereIntersectNearest(
    worldPosition, worldDirection, planet0, atmosphereParameters.top_radius);
    float integrationStep = integrationLength / float(sampleCount);
    /* Result of the integration */
    vec3 opticalDepth = vec3(0.0,0.0,0.0);

    for(int i = 0; i < sampleCount; i++)
    {
        /* Move along the world direction ray to new position */
        vec3 newPos = worldPosition + i * integrationStep * worldDirection;
        vec3 atmosphereExtinction = SampleMediumExtinction(newPos);
        opticalDepth += atmosphereExtinction * integrationStep;
    }
    return opticalDepth;
}

void main()
{
    vec2 atmosphereBoundaries = vec2(
    atmosphereParameters.bottom_radius,
    atmosphereParameters.top_radius);

    vec2 uv = gl_GlobalInvocationID.xy / atmosphereParameters.TransmittanceTexDimensions;
    vec2 LUTParams = UvToTransmittanceLUTParams(uv, atmosphereBoundaries);

    /* Ray origin in World Coordinates */
    vec3 worldPosition = vec3(0.0, 0.0, LUTParams.x);
    /* Ray direction in World Coordinates */
    vec3 worldDirection = vec3(0.0, safeSqrt(1.0 - LUTParams.y * LUTParams.y), LUTParams.y);
    vec3 transmittance = exp(-IntegrateTransmittance(worldPosition, worldDirection, 400));
    imageStore(transmittanceLUT, ivec2(gl_GlobalInvocationID.xy), vec4(transmittance, 1.0));
}

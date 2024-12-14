#version 450

layout (local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

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
layout (binding = 4, rgba16f) uniform readonly image2D multiscatteringLUT;
layout (binding = 5, rgba16f) uniform readonly image2D skyViewLUT;
layout (binding = 6, rgba16f) uniform image3D AEPerspective;

/* One unit in global space should be 100 meters in camera coords */
const float cameraScale = 0.1;
struct ScatteringSample
{
    vec3 Mie;
    vec3 Ray;
};

/* ============================= PHASE FUNCTIONS ============================ */
float cornetteShanksMiePhaseFunction(float g, float cosTheta)
{
    float k = 3.0 / (8.0 * PI) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + cosTheta * cosTheta) / pow(1.0 + g * g - 2.0 * g * -cosTheta, 1.5);
}

float rayleighPhase(float cosTheta)
{
    float factor = 3.0 / (16.0 * PI);
    return factor * (1.0 + cosTheta * cosTheta);
}
/* ========================================================================== */

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

ScatteringSample SampleMediumScattering(vec3 worldPosition)
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

    return ScatteringSample(mieScattering, rayleighScattering);
}
/* ========================================================================== */

vec3 getMultipleScattering(vec3 worldPosition, float viewZenithCosAngle)
{
    vec2 uv = clamp(vec2(
    viewZenithCosAngle * 0.5 + 0.5,
    (length(worldPosition) - atmosphereParameters.bottom_radius) /
    (atmosphereParameters.top_radius - atmosphereParameters.bottom_radius)),
    0.0, 1.0);
    uv = vec2(fromUnitToSubUvs(uv.x, atmosphereParameters.MultiscatteringTexDimensions.x),
    fromUnitToSubUvs(uv.y, atmosphereParameters.MultiscatteringTexDimensions.y));
    ivec2 coords = ivec2(uv * atmosphereParameters.MultiscatteringTexDimensions);
    return imageLoad(multiscatteringLUT, coords).rgb;
}

struct RaymarchResult
{
    vec3 Luminance;
    vec3 Transmittance;
};

RaymarchResult integrateScatteredLuminance(vec3 worldPosition, vec3 worldDirection,
vec3 sunDirection, int sampleCount, float maxDist)
{
    RaymarchResult result = RaymarchResult(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0));
    vec2 atmosphereBoundaries = vec2(atmosphereParameters.bottom_radius, atmosphereParameters.top_radius);

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
    integrationLength = min(integrationLength, maxDist);
    float cosTheta = dot(sunDirection, worldDirection);
    float miePhaseValue = cornetteShanksMiePhaseFunction(atmosphereParameters.mie_phase_function_g, -cosTheta);
    float rayleighPhaseValue = rayleighPhase(cosTheta);
    float oldRayShift = 0.0;
    float integrationStep = 0.0;

    vec3 accumTrans = vec3(1.0, 1.0, 1.0);
    vec3 accumLight = vec3(0.0, 0.0, 0.0);
    /* ============================= RAYMARCH ============================ */
    for(int i = 0; i < sampleCount; i++)
    {
        float newRayShift = integrationLength * (float(i) + 0.3) / sampleCount;
        integrationStep = newRayShift - oldRayShift;
        vec3 newPos = worldPosition + newRayShift * worldDirection;
        oldRayShift = newRayShift;

        ScatteringSample mediumScattering = SampleMediumScattering(newPos);
        vec3 mediumExtinction = SampleMediumExtinction(newPos);

        /* Raymarch shifts the angle to the sun a bit recalculate */
        vec3 upVector = normalize(newPos);
        vec2 transLUTParams = vec2( length(newPos),dot(sunDirection, upVector));

        /* uv coordinates later used to sample transmittance texture */
        vec2 transUV = TransmittanceLUTParamsToUv(transLUTParams, atmosphereBoundaries);
        /* because here transmittanceLUT is image and not a texture transfer
           from [0,1] -> [tex_width, tex_height] */
        ivec2 transImageCoords = ivec2(transUV * atmosphereParameters.TransmittanceTexDimensions);

        vec3 transmittanceToSun = vec3(imageLoad(transmittanceLUT, transImageCoords).rgb);
        vec3 phaseTimesScattering = mediumScattering.Mie * miePhaseValue +
        mediumScattering.Ray * rayleighPhaseValue;

        float earthIntersectionDistance = raySphereIntersectNearest(
        newPos, sunDirection, planet0 + PLANET_RADIUS_OFFSET * upVector, atmosphereParameters.bottom_radius);
        float inEarthShadow = earthIntersectionDistance == -1.0 ? 1.0 : 0.0;

        vec3 multiscatteredLuminance = getMultipleScattering(newPos, dot(sunDirection, upVector));

        /* Light arriving from the sun to this point */
        vec3 sunLight = inEarthShadow * transmittanceToSun * phaseTimesScattering +
        multiscatteredLuminance * (mediumScattering.Ray + mediumScattering.Mie);

        /* TODO: This probably should be a texture lookup*/
        vec3 transIncreseOverInegrationStep = exp(-(mediumExtinction * integrationStep));
        vec3 sunLightInteg = (sunLight - sunLight * transIncreseOverInegrationStep) / mediumExtinction;
        accumLight += accumTrans * sunLightInteg;
        accumTrans *= transIncreseOverInegrationStep;
    }
    result.Luminance = accumLight;
    result.Transmittance = accumTrans;

    return result;
}

void main()
{

    vec3 camera = atmosphereParameters.camera_position.xyz;
    vec3 sun_direction = atmosphereParameters.sun_direction.xyz;

    mat4 invViewProjMat = inverse(commonParameters.lHviewProj);
    invViewProjMat = inverse(commonParameters.proj * commonParameters.view);
    vec2 pixPos = vec2(gl_GlobalInvocationID.xy + vec2(0.5, 0.5)) / atmosphereParameters.AEPerspectiveTexDimensions.xy;
    vec3 ClipSpace = vec3(pixPos*vec2(2.0, 2.0) - vec2(1.0, 1.0), 0.5);

    vec4 Hpos = invViewProjMat * vec4(ClipSpace, 1.0);
    vec3 worldDirection = normalize(Hpos.xyz / Hpos.w - camera);
    vec3 cameraPosition = camera  * cameraScale + vec3(0.0, 0.0, atmosphereParameters.bottom_radius);

    float Slice = ((float(gl_GlobalInvocationID.z) + 0.5) / atmosphereParameters.AEPerspectiveTexDimensions.z);
    Slice *= Slice;
    Slice *= atmosphereParameters.AEPerspectiveTexDimensions.z;

    /* TODO: Change slice size to be uniform variable */
    float tMax = Slice * 4.0;
    vec3 newWorldPos = cameraPosition + tMax * worldDirection;


    float viewHeight = length(newWorldPos);
    if (viewHeight <= (atmosphereParameters.bottom_radius + PLANET_RADIUS_OFFSET))
    {
        newWorldPos = normalize(newWorldPos) * (atmosphereParameters.bottom_radius + PLANET_RADIUS_OFFSET + 0.001);
        worldDirection = normalize(newWorldPos - cameraPosition);
        tMax = length(newWorldPos - cameraPosition);
    }

    viewHeight = length(cameraPosition);
    vec2 atmosphereBoundaries = vec2(atmosphereParameters.bottom_radius, atmosphereParameters.top_radius);
    if(viewHeight >= atmosphereParameters.top_radius)
    {
        vec3 prevWorldPos = cameraPosition;
        if(!moveToTopAtmosphere(cameraPosition, worldDirection, atmosphereBoundaries))
        {
            imageStore(AEPerspective, ivec3(gl_GlobalInvocationID.xyz), vec4( 0.0, 0.0, 0.0, 1.0));
            return;
        }
        float lengthToAtmosphere = length(prevWorldPos - cameraPosition);
        if(tMax < lengthToAtmosphere)
        {
            imageStore(AEPerspective, ivec3(gl_GlobalInvocationID.xyz), vec4( 0.0, 0.0, 0.0, 1.0));
            return;
        }
        tMax = max(0.0, tMax - lengthToAtmosphere);
    }
    int sampleCount = int(max(1.0, float(gl_GlobalInvocationID.z + 1.0) * 2.0));
    RaymarchResult res = integrateScatteredLuminance(cameraPosition, worldDirection, sun_direction,
    sampleCount, tMax);
    float averageTransmittance = (res.Transmittance.x + res.Transmittance.y + res.Transmittance.z) / 3.0;

    imageStore(AEPerspective, ivec3(gl_GlobalInvocationID.xyz), vec4(res.Luminance, averageTransmittance));
}

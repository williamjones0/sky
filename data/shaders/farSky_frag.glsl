/* Parts of code take from https://github.com/SebLague/Clouds/blob/master/Assets/Scripts/Clouds/Shaders/Clouds.shader */
#version 450

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


layout (location = 0) out vec4 outColor;
layout (location = 0) in vec2 inUV;

layout(binding = 0) uniform CommonParamBufferObject
{
    mat4 model;
    mat4 view;
    mat4 proj;
    mat4 lHviewProj;
    float time;
//    float test0;
//    float test1;
//    float test2;
//    float test3;
//    float test4;
//    float test5;
//    float test6;
//    float test7;
//    float test8;
//    float test9;
//    float test10;
//    float test11;
//    float test12;
//    float test13;
//    float test14;
//    float test15;
//    float test16;
//
//    float test17;
//    float test18;
//    float test19;
//    float test20;
//    float test21;
//    float test22;
//    float test23;
//    float test24;
//    float test25;
//    float test26;
//    float test27;
//    float test28;
//    float test29;
//    float test30;
//    float test31;
//    float test32;
//
//    float test33;
//    float test34;
//    float test35;
//    float test36;
//    float test37;
//    float test38;
//    float test39;
//    float test40;
//    float test41;
//    float test42;
//    float test43;
//    float test44;
//    float test45;
//    float test46;
//    float test47;
//    float test48;
//
//    float test49;
//    float test50;
//    float test51;
//    float test52;
//    float test53;
//    float test54;
//    float test55;
//    float test56;
//    float test57;
//    float test58;
//    float test59;
//    float test60;
//    float test61;
//    float test62;
//    float test63;
//    float test64;
//
//    float time;
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

layout (binding = 5) uniform sampler2D texSampler;
//layout (binding = 7, r32f) uniform readonly image2D depthInput;

vec3 sunWithBloom(vec3 worldDir, vec3 sunDir)
{
    const float sunSolidAngle = 1.0 * PI / 180.0;
    const float minSunCosTheta = cos(sunSolidAngle);

    float cosTheta = dot(worldDir, sunDir);
    if(cosTheta >= minSunCosTheta) {return vec3(0.5) ;}
    float offset = minSunCosTheta - cosTheta;
    float gaussianBloom = exp(-offset * 50000.0) * 0.5;
    float invBloom = 1.0/(0.02 + offset * 300.0) * 0.01;
    return vec3(gaussianBloom + invBloom);
}
/* One unit in global space should be 100 meters in camera coords */
const float cameraScale = 0.1;
void main()
{
    const vec3 camera = atmosphereParameters.camera_position.xyz;
    vec3 sun_direction = atmosphereParameters.sun_direction.xyz;
    vec2 atmosphereBoundaries = vec2(
    atmosphereParameters.bottom_radius,
    atmosphereParameters.top_radius);

    mat4 invViewProjMat = inverse(commonParameters.proj * commonParameters.view);
    vec2 pixPos = inUV.yx;
    vec3 ClipSpace = vec3(pixPos*vec2(2.0) - vec2(1.0), 1.0);

    vec4 Hpos = invViewProjMat * vec4(ClipSpace, 1.0);

    vec3 WorldDir = normalize(Hpos.xyz/Hpos.w - camera);
    vec3 WorldPos = camera * cameraScale + vec3(0,0, atmosphereParameters.bottom_radius);

    float viewHeight = length(WorldPos);
    vec3 L = vec3(0.0,0.0,0.0);

//    float depth = subpassLoad(depthInput).r;
//    float depth = imageLoad(depthInput, ivec2(gl_FragCoord.xy)).r;
    float depth = 1.0;

    if(depth == 1.0)
    {
        vec2 uv;
        vec3 UpVector = normalize(WorldPos);
        float viewZenithAngle = acos(dot(WorldDir, UpVector));

        float lightViewAngle = acos(dot(normalize(vec3(sun_direction.xy, 0.0)), normalize(vec3(WorldDir.xy, 0.0))));
        bool IntersectGround = raySphereIntersectNearest(WorldPos, WorldDir, vec3(0.0, 0.0, 0.0),
        atmosphereParameters.bottom_radius) >= 0.0;

        uv = SkyViewLutParamsToUv(IntersectGround, vec2(viewZenithAngle,lightViewAngle), viewHeight,
        atmosphereBoundaries, atmosphereParameters.SkyViewTexDimensions);

        L += vec3(texture(texSampler, vec2(uv.x, uv.y)).rgb);

        if(!IntersectGround)
        {
            L += sunWithBloom(WorldDir, sun_direction);
        };

        outColor = vec4(L, 1.0);
    }
    else{
        outColor = vec4(0.0, 0.0, 0.0, 0.0);
    }
}

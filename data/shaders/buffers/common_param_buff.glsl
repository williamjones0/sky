layout(set = 0, binding = 0) uniform CommonParamBufferObject
{
    mat4 model;
    mat4 view;
    mat4 proj;
	mat4 lHviewProj;
    float time;
} commonParameters;
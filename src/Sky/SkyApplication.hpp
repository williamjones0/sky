#pragma once

#include "Application.hpp"

#include "core/Camera.hpp"
#include "model/SkyModel.hpp"
#include "opengl/BufferDefines.hpp"
#include "opengl/Shader.hpp"

class SkyApplication final : public Application {
public:
    void key_callback(int key, int scancode, int action, int mods) override;
    void mouse_button_callback(int button, int action, int mods) override;
    void cursor_position_callback(double xposIn, double yposIn) override;
    void scroll_callback(double xoffset, double yoffset) override;

protected:
    bool init() override;
    bool load() override;
    void update() override;
    void render() override;
    void processInput() override;
    void cleanup() override;

private:
    void updateUniformBuffer();
    void renderLUTs();

    Camera camera;

    int transmittanceWidth = 256;
    int transmittanceHeight = 64;
    int scatteringWidth = 32;
    int scatteringHeight = 32;
    int skyViewWidth = 192;
    int skyViewHeight = 128;
    int aerialPerspectiveWidth = 32;
    int aerialPerspectiveHeight = 32;
    int aerialPerspectiveDepth = 32;

    Shader transmittanceShader;
    Shader multiScatteringShader;
    Shader skyViewShader;
    Shader aerialPerspectiveShader;
    Shader skyShader;
    Shader histogramGenerateShader;
    Shader histogramSumShader;
    Shader finalCompositionShader;

    GLuint uboHandle;
    GLuint atmosphereParametersBufferObject;
    GLuint postProcessParametersBufferObject;
    GLuint histogram;
    GLuint averageLuminance;
    GLuint transmittance;
    GLuint scattering;
    GLuint skyView;
    GLuint aerialPerspective;

    GLuint framebuffer;
    GLuint framebufferColor;
    GLuint framebufferDepth;

    GLuint VAO;

    AtmosphereParametersBuffer atmoParamsBuffer;
    PostProcessParamsBuffer postProcessParamsBuffer;
};

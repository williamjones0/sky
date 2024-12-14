#include "SkyApplication.hpp"
#include "glm/gtx/string_cast.hpp"

bool SkyApplication::init() {
    if (!Application::init()) {
        return false;
    }

    glfwSetWindowUserPointer(windowHandle, this);

    glfwSetKeyCallback(windowHandle, [](GLFWwindow *window, int key, int scancode, int action, int mods) {
        auto *app = static_cast<SkyApplication *>(glfwGetWindowUserPointer(window));
        app->key_callback(key, scancode, action, mods);
    });
    glfwSetMouseButtonCallback(windowHandle, [](GLFWwindow *window, int button, int action, int mods) {
        auto *app = static_cast<SkyApplication *>(glfwGetWindowUserPointer(window));
        app->mouse_button_callback(button, action, mods);
    });
    glfwSetCursorPosCallback(windowHandle, [](GLFWwindow *window, double xpos, double ypos) {
        auto *app = static_cast<SkyApplication *>(glfwGetWindowUserPointer(window));
        app->cursor_position_callback(xpos, ypos);
    });
    glfwSetScrollCallback(windowHandle, [](GLFWwindow *window, double xoffset, double yoffset) {
        auto *app = static_cast<SkyApplication *>(glfwGetWindowUserPointer(window));
        app->scroll_callback(xoffset, yoffset);
    });

    return true;
}

bool SkyApplication::load() {
    if (!Application::load()) {
        return false;
    }

    // Textures
    glCreateTextures(GL_TEXTURE_2D, 1, &transmittance);
    glTextureParameteri(transmittance, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(transmittance, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTextureParameteri(transmittance, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(transmittance, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureStorage2D(transmittance, 1, GL_RGBA16F, transmittanceWidth, transmittanceHeight);
    glBindImageTexture(3, transmittance, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);

    glCreateTextures(GL_TEXTURE_2D, 1, &scattering);
    glTextureParameteri(scattering, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(scattering, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTextureParameteri(scattering, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(scattering, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureStorage2D(scattering, 1, GL_RGBA16F, scatteringWidth, scatteringHeight);
    glBindImageTexture(4, scattering, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);

    glCreateTextures(GL_TEXTURE_2D, 1, &skyView);
    glTextureParameteri(skyView, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(skyView, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTextureParameteri(skyView, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(skyView, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureStorage2D(skyView, 1, GL_RGBA16F, skyViewWidth, skyViewHeight);
    glBindImageTexture(5, skyView, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);

    glCreateTextures(GL_TEXTURE_3D, 1, &aerialPerspective);
    glTextureParameteri(aerialPerspective, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(aerialPerspective, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTextureParameteri(aerialPerspective, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTextureParameteri(aerialPerspective, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(aerialPerspective, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureStorage3D(aerialPerspective, 1, GL_RGBA16F, aerialPerspectiveWidth, aerialPerspectiveHeight, aerialPerspectiveDepth);
    glBindImageTexture(6, aerialPerspective, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);

    // Framebuffer
    glCreateTextures(GL_TEXTURE_2D, 1, &framebufferColor);
    glTextureParameteri(framebufferColor, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(framebufferColor, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTextureParameteri(framebufferColor, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(framebufferColor, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureStorage2D(framebufferColor, 1, GL_RGBA32F, windowWidth, windowHeight);
    glBindImageTexture(10, framebufferColor, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);

    glCreateTextures(GL_TEXTURE_2D, 1, &framebufferDepth);
    glTextureParameteri(framebufferDepth, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(framebufferDepth, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTextureParameteri(framebufferDepth, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(framebufferDepth, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTextureStorage2D(framebufferDepth, 1, GL_DEPTH_COMPONENT32F, windowWidth, windowHeight);
    glBindImageTexture(7, framebufferDepth, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);

    glCreateFramebuffers(1, &framebuffer);
    glNamedFramebufferTexture(framebuffer, GL_COLOR_ATTACHMENT0, framebufferColor, 0);
    glNamedFramebufferTexture(framebuffer, GL_DEPTH_ATTACHMENT, framebufferDepth, 0);

    // Shaders
    transmittanceShader = Shader("transmittance_comp.glsl");
    multiScatteringShader = Shader("multiScattering_comp.glsl");
    skyViewShader = Shader("skyView_comp.glsl");
    aerialPerspectiveShader = Shader("aerialPerspective_comp.glsl");
    skyShader = Shader("quad_vert.glsl", "farSky_frag.glsl");
    histogramGenerateShader = Shader("histogramGenerate_comp.glsl");
    histogramSumShader = Shader("histogramSum_comp.glsl");
    finalCompositionShader = Shader("quad_vert.glsl", "finalComposition_frag.glsl");

    // Screen quad
    float vertices[] = {
            -1.0f, 1.0f, 0.0f,   // top left
            -1.0f, -1.0f, 0.0f,  // bottom left
            1.0f, -1.0f, 0.0f,   // bottom right
            1.0f, 1.0f, 0.0f,   // top right
    };

    float texCoords[] = {
            0.0f, 1.0f,
            0.0f, 0.0f,
            1.0f, 0.0f,
            1.0f, 1.0f,
    };

    int indices[] = {
            0, 1, 3,
            3, 1, 2
    };

    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    GLuint positionsVBO;
    glCreateBuffers(1, &positionsVBO);
    glBindBuffer(GL_ARRAY_BUFFER, positionsVBO);
    glNamedBufferStorage(positionsVBO, sizeof(vertices), vertices, 0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);

    GLuint texCoordsVBO;
    glCreateBuffers(1, &texCoordsVBO);
    glBindBuffer(GL_ARRAY_BUFFER, texCoordsVBO);
    glNamedBufferStorage(texCoordsVBO, sizeof(texCoords), texCoords, 0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
    glEnableVertexAttribArray(1);

    GLuint indicesEBO;
    glCreateBuffers(1, &indicesEBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indicesEBO);
    glNamedBufferStorage(indicesEBO, sizeof(indices), indices, 0);

    // Buffers
    glCreateBuffers(1, &uboHandle);
    glNamedBufferStorage(
            uboHandle,
            sizeof(UniformBufferObject),
            nullptr,
            GL_DYNAMIC_STORAGE_BIT
    );
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, uboHandle);

    glCreateBuffers(1, &atmosphereParametersBufferObject);
    glNamedBufferStorage(
            atmosphereParametersBufferObject,
            sizeof(AtmosphereParametersBuffer),
            nullptr,
            GL_DYNAMIC_STORAGE_BIT
    );
    glBindBufferBase(GL_UNIFORM_BUFFER, 1, atmosphereParametersBufferObject);

    glCreateBuffers(1, &postProcessParametersBufferObject);
    glNamedBufferStorage(
            postProcessParametersBufferObject,
            sizeof(PostProcessParamsBuffer),
            nullptr,
            GL_DYNAMIC_STORAGE_BIT
    );
    glBindBufferBase(GL_UNIFORM_BUFFER, 2, postProcessParametersBufferObject);

    int jef0 = sizeof(UniformBufferObject);
    int jef1 = sizeof(AtmosphereParametersBuffer);
    int jef2 = sizeof(PostProcessParamsBuffer);

    // binding index 3 is reserved for clouds later on

    glCreateBuffers(1, &histogram);
    glNamedBufferStorage(
            histogram,
            sizeof(uint32_t) * 256,
            nullptr,
            GL_DYNAMIC_STORAGE_BIT
    );
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, histogram);

    glCreateBuffers(1, &averageLuminance);
    glNamedBufferStorage(
            averageLuminance,
            sizeof(float),
            nullptr,
            GL_DYNAMIC_STORAGE_BIT
    );
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, averageLuminance);

    // Camera
    camera = Camera(glm::vec3(0.0f, 0.0f, 0.2f), -90.0f, 0.0f);

    return true;
}

void SkyApplication::update() {
    Application::update();

    updateUniformBuffer();
}

void SkyApplication::render() {
    renderLUTs();

    // Sky
    glViewport(0, 0, windowWidth, windowHeight);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    skyShader.use();
    glUniform1i(glGetUniformLocation(skyShader.ID, "texSampler"), 5);
    glUniform1i(glGetUniformLocation(skyShader.ID, "depthInput"), 6);
    glBindTextureUnit(5, skyView);
    glBindTextureUnit(7, framebufferDepth);
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

    // Histogram generation
    histogramGenerateShader.use();
    glUniform1i(glGetUniformLocation(histogramGenerateShader.ID, "histogram"), 8);
    glUniform1i(glGetUniformLocation(histogramGenerateShader.ID, "averageLuminance"), 9);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, histogram);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, averageLuminance);
    glDispatchCompute(skyViewWidth / 16, skyViewHeight / 16, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // Histogram sum
    histogramSumShader.use();
    glUniform1i(glGetUniformLocation(histogramSumShader.ID, "histogram"), 8);
    glUniform1i(glGetUniformLocation(histogramSumShader.ID, "averageLuminance"), 9);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, histogram);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, averageLuminance);
    glDispatchCompute(1, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // Final composition (bind default framebuffer)
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    finalCompositionShader.use();
    glUniform1i(glGetUniformLocation(finalCompositionShader.ID, "texSampler"), 10);
    glUniform1i(glGetUniformLocation(finalCompositionShader.ID, "averageLuminance"), 9);
    glBindTextureUnit(10, framebufferColor);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, averageLuminance);
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
}

void SkyApplication::processInput() {
    for (int key : keys) {
        switch (key) {
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(windowHandle, true);
                break;

            case GLFW_KEY_W:
                camera.processKeyboard(FORWARD, deltaTime);
                break;
            case GLFW_KEY_S:
                camera.processKeyboard(BACKWARD, deltaTime);
                break;
            case GLFW_KEY_A:
                camera.processKeyboard(LEFT, deltaTime);
                break;
            case GLFW_KEY_D:
                camera.processKeyboard(RIGHT, deltaTime);
                break;
            case GLFW_KEY_SPACE:
                camera.processKeyboard(UP, deltaTime);
                break;
            case GLFW_KEY_LEFT_SHIFT:
                camera.processKeyboard(DOWN, deltaTime);
                break;
            default:
                break;
        }
    }

    Application::processInput();
}

void SkyApplication::cleanup() {
    Application::cleanup();
}

void SkyApplication::updateUniformBuffer() {
    SetupAtmosphereParametersBuffer(atmoParamsBuffer);

    postProcessParamsBuffer = PostProcessParamsBuffer{};
    postProcessParamsBuffer.texDimensions = glm::vec2(windowWidth, windowHeight);
    postProcessParamsBuffer.minimumLuminance = 100.0;
    postProcessParamsBuffer.maximumLuminance = 6000.0;
    postProcessParamsBuffer.lumAdaptTau = 1.1;
    postProcessParamsBuffer.tonemapCurve = 4;
    postProcessParamsBuffer.whitepoint = 4.0;
    postProcessParamsBuffer.maxDisplayBrigtness = 1.0;
    postProcessParamsBuffer.contrast = 1.0;
    postProcessParamsBuffer.linearSectionLength = 0.22;
    postProcessParamsBuffer.linearSectionLength = 0.4;
    postProcessParamsBuffer.black = 1.33;
    postProcessParamsBuffer.pedestal = 0.0;
    postProcessParamsBuffer.a = 1.6;
    postProcessParamsBuffer.d = 0.977;
    postProcessParamsBuffer.hdrMax = 8.0;
    postProcessParamsBuffer.midIn = 0.18;
    postProcessParamsBuffer.midOut = 0.267;

    UniformBufferObject ubo{};
    ubo.model = glm::mat4(1.0f);
    ubo.model = glm::scale(ubo.model, glm::vec3(1000, 1000, 1000.0));
    ubo.model = glm::translate(ubo.model, glm::vec3(-0.5f, -0.5f, -0.0f));

    float aspectRatio = static_cast<float>(windowWidth) / static_cast<float>(windowHeight);
    ubo.proj = glm::perspective(glm::radians(50.0f), aspectRatio, 0.1f, 20000.0f);

    ubo.view = camera.calculateViewMatrix();
    ubo.lHviewProj = ubo.proj * camera.calculateViewMatrix(true);
    ubo.time = 20.0f;

//    // create view matrix with direction = vec3(0.0, -1.0, 0.5)
//    glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 0.2f);
//    glm::vec3 cameraFront = glm::vec3(0.0f, -1.0f, 0.5f);
//    glm::vec3 cameraUp = glm::vec3(0.0f, 0.0f, 1.0f);
//    ubo.view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
//    ubo.lHviewProj = ubo.proj * glm::lookAtLH(cameraPos, cameraPos + cameraFront, cameraUp);
//    ubo.time = 20.0f;

    glNamedBufferSubData(uboHandle, 0, sizeof(UniformBufferObject), &ubo);

    glNamedBufferSubData(atmosphereParametersBufferObject, 0, sizeof(AtmosphereParametersBuffer), &atmoParamsBuffer);

    glNamedBufferSubData(postProcessParametersBufferObject, 0, sizeof(PostProcessParamsBuffer), &postProcessParamsBuffer);
}

void SkyApplication::renderLUTs() {
    // Transmittance
    transmittanceShader.use();
    glUniform1i(glGetUniformLocation(transmittanceShader.ID, "transmittanceLUT"), 3);
    glBindTextureUnit(3, transmittance);
    glDispatchCompute(transmittanceWidth / 8, transmittanceHeight / 4, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    // Multi-scattering
    multiScatteringShader.use();
    glUniform1i(glGetUniformLocation(multiScatteringShader.ID, "transmittanceLUT"), 3);
    glUniform1i(glGetUniformLocation(multiScatteringShader.ID, "multiscatteringLUT"), 4);
    glBindTextureUnit(3, transmittance);
    glBindTextureUnit(4, scattering);
    glDispatchCompute(32, 32, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    // Sky view
    skyViewShader.use();
    glUniform1i(glGetUniformLocation(skyViewShader.ID, "transmittanceLUT"), 3);
    glUniform1i(glGetUniformLocation(skyViewShader.ID, "multiscatteringLUT"), 4);
    glUniform1i(glGetUniformLocation(skyViewShader.ID, "skyViewLUT"), 5);
    glBindTextureUnit(3, transmittance);
    glBindTextureUnit(4, scattering);
    glBindTextureUnit(5, skyView);
    glDispatchCompute(skyViewWidth / 16, skyViewHeight / 16, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    // Aerial perspective
    aerialPerspectiveShader.use();
    glUniform1i(glGetUniformLocation(aerialPerspectiveShader.ID, "transmittanceLUT"), 3);
    glUniform1i(glGetUniformLocation(aerialPerspectiveShader.ID, "multiscatteringLUT"), 4);
    glUniform1i(glGetUniformLocation(aerialPerspectiveShader.ID, "skyViewLUT"), 5);
    glUniform1i(glGetUniformLocation(aerialPerspectiveShader.ID, "AEPerspective"), 6);
    glBindTextureUnit(3, transmittance);
    glBindTextureUnit(4, scattering);
    glBindTextureUnit(5, skyView);
    glBindTextureUnit(6, aerialPerspective);
    glDispatchCompute(1, 32, 32);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

void SkyApplication::key_callback(int key, int scancode, int action, int mods) {
    Application::key_callback(key, scancode, action, mods);
}

void SkyApplication::mouse_button_callback(int button, int action, int mods) {
    Application::mouse_button_callback(button, action, mods);
}

void SkyApplication::cursor_position_callback(double xposIn, double yposIn) {
    auto xpos = static_cast<float>(xposIn);
    auto ypos = static_cast<float>(yposIn);

    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;

    lastX = xpos;
    lastY = ypos;

    camera.processMouse(xoffset, yoffset, 0);
}

void SkyApplication::scroll_callback(double xoffset, double yoffset) {
    camera.processMouse(0, 0, yoffset);
}

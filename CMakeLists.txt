cmake_minimum_required(VERSION 3.24)

project(vk_cs LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Vulkan REQUIRED)

set(SDL3_ROOT "D:/tools/SDL3-3.4.2" CACHE PATH "Path to SDL3 installation root")
set(SDL3_INCLUDE_DIR "${SDL3_ROOT}/include")
set(SDL3_LIBRARY "${SDL3_ROOT}/lib/x64/SDL3.lib")
set(SDL3_DLL "${SDL3_ROOT}/lib/x64/SDL3.dll")
if(NOT EXISTS "${SDL3_INCLUDE_DIR}/SDL3/SDL.h")
    message(FATAL_ERROR "SDL3 headers not found at ${SDL3_INCLUDE_DIR}")
endif()
if(NOT EXISTS "${SDL3_LIBRARY}")
    message(FATAL_ERROR "SDL3 library not found at ${SDL3_LIBRARY}")
endif()

find_program(
    GLSLC_EXECUTABLE
    NAMES glslc
    HINTS
        "$ENV{VULKAN_SDK}/Bin"
        "$ENV{VULKAN_SDK}/Bin32"
)

if(NOT GLSLC_EXECUTABLE)
    message(FATAL_ERROR "glslc not found. Install Vulkan SDK and ensure glslc is in PATH.")
endif()

set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADER_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_OUT_DIR})

function(compile_shader SHADER_FILE OUT_LIST)
    get_filename_component(SHADER_NAME ${SHADER_FILE} NAME)
    set(SPV_FILE ${SHADER_OUT_DIR}/${SHADER_NAME}.spv)
    add_custom_command(
        OUTPUT ${SPV_FILE}
        COMMAND ${GLSLC_EXECUTABLE} --target-env=vulkan1.2 -O ${SHADER_FILE} -o ${SPV_FILE}
        DEPENDS ${SHADER_FILE}
        COMMENT "Compiling shader ${SHADER_NAME}"
        VERBATIM
    )
    set(${OUT_LIST} ${${OUT_LIST}} ${SPV_FILE} PARENT_SCOPE)
endfunction()

set(SHADER_SPV)
compile_shader(${SHADER_DIR}/gradient_tet_stage.comp SHADER_SPV)
compile_shader(${SHADER_DIR}/gradient_vertex_gather.comp SHADER_SPV)
compile_shader(${SHADER_DIR}/hessp_tet_stage.comp SHADER_SPV)
compile_shader(${SHADER_DIR}/hessp_vertex_gather.comp SHADER_SPV)
compile_shader(${SHADER_DIR}/predict_state.comp SHADER_SPV)
compile_shader(${SHADER_DIR}/update_velocity_state.comp SHADER_SPV)
compile_shader(${SHADER_DIR}/reconstruct_x_from_reduced.comp SHADER_SPV)
compile_shader(${SHADER_DIR}/build_world_from_reduced.comp SHADER_SPV)
compile_shader(${SHADER_DIR}/project_world_to_reduced_stage1.comp SHADER_SPV)
compile_shader(${SHADER_DIR}/project_world_to_reduced_stage2.comp SHADER_SPV)
compile_shader(${SHADER_DIR}/reduced_cg_init.comp SHADER_SPV)
compile_shader(${SHADER_DIR}/reduced_cg_update.comp SHADER_SPV)

add_custom_target(vk_cs_shaders DEPENDS ${SHADER_SPV})

add_executable(vk_cs
    src/main.cpp
    src/mesh_loader.cpp
    src/subspace_loader.cpp
    src/csr_builder.cpp
    src/vulkan_context.cpp
    src/vk_two_phase.cpp
    src/subspace_simulator.cpp
    src/simple_renderer.cpp
)

target_include_directories(vk_cs PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/libs
    ${SDL3_INCLUDE_DIR}
)

target_link_libraries(vk_cs PRIVATE Vulkan::Vulkan "${SDL3_LIBRARY}")
if(WIN32)
    if(EXISTS "${SDL3_DLL}")
        add_custom_command(TARGET vk_cs POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${SDL3_DLL}"
                $<TARGET_FILE_DIR:vk_cs>
        )
    endif()
endif()
add_dependencies(vk_cs vk_cs_shaders)

target_compile_definitions(vk_cs PRIVATE
    VKCS_SHADER_DIR="${SHADER_OUT_DIR}"
)

/*
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#ifndef AVFILTER_VULKAN_H
#define AVFILTER_VULKAN_H

#define VK_NO_PROTOTYPES
#define VK_ENABLE_BETA_EXTENSIONS

#include "avfilter.h"
#include "libavutil/pixdesc.h"
#include "libavutil/bprint.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_vulkan.h"
#include "libavutil/vulkan_functions.h"

/* GLSL management macros */
#define INDENT(N) INDENT_##N
#define INDENT_0
#define INDENT_1 INDENT_0 "    "
#define INDENT_2 INDENT_1 INDENT_1
#define INDENT_3 INDENT_2 INDENT_1
#define INDENT_4 INDENT_3 INDENT_1
#define INDENT_5 INDENT_4 INDENT_1
#define INDENT_6 INDENT_5 INDENT_1
#define C(N, S)          INDENT(N) #S "\n"
#define GLSLC(N, S)      av_bprintf(&shd->src, C(N, S))
#define GLSLA(...)       av_bprintf(&shd->src, __VA_ARGS__)
#define GLSLF(N, S, ...) av_bprintf(&shd->src, C(N, S), __VA_ARGS__)
#define GLSLD(D)         GLSLC(0, );                                           \
                         av_bprint_append_data(&shd->src, D, strlen(D));       \
                         GLSLC(0, )

/* Helper, pretty much every Vulkan return value needs to be checked */
#define RET(x)                                                                 \
    do {                                                                       \
        if ((err = (x)) < 0)                                                   \
            goto fail;                                                         \
    } while (0)

typedef struct FFSPIRVShader {
    const char *name;                       /* Name for id/debugging purposes */
    AVBPrint src;
    int local_size[3];                      /* Compute shader workgroup sizes */
    VkPipelineShaderStageCreateInfo shader;
} FFSPIRVShader;

typedef struct FFVkSampler {
    VkSampler sampler[4];
} FFVkSampler;

typedef struct FFVulkanDescriptorSetBinding {
    const char         *name;
    VkDescriptorType    type;
    const char         *mem_layout;  /* Storage images (rgba8, etc.) and buffers (std430, etc.) */
    const char         *mem_quali;   /* readonly, writeonly, etc. */
    const char         *buf_content; /* For buffers */
    uint32_t            dimensions;  /* Needed for e.g. sampler%iD */
    uint32_t            elems;       /* 0 - scalar, 1 or more - vector */
    VkShaderStageFlags  stages;
    FFVkSampler        *sampler;     /* Sampler to use for all elems */
    void               *updater;     /* Pointer to VkDescriptor*Info */
} FFVulkanDescriptorSetBinding;

typedef struct FFVkBuffer {
    VkBuffer buf;
    VkDeviceMemory mem;
    VkMemoryPropertyFlagBits flags;
} FFVkBuffer;

typedef struct FFVkQueueFamilyCtx {
    int queue_family;
    int nb_queues;
    int cur_queue;
} FFVkQueueFamilyCtx;

typedef struct FFVulkanPipeline {
    FFVkQueueFamilyCtx *qf;

    VkPipelineBindPoint bind_point;

    /* Contexts */
    VkPipelineLayout pipeline_layout;
    VkPipeline       pipeline;

    /* Shaders */
    FFSPIRVShader **shaders;
    int shaders_num;

    /* Push consts */
    VkPushConstantRange *push_consts;
    int push_consts_num;

    /* Descriptors */
    VkDescriptorSetLayout         *desc_layout;
    VkDescriptorPool               desc_pool;
    VkDescriptorSet               *desc_set;
    void                         **desc_staging;
    VkDescriptorSetLayoutBinding **desc_binding;
    VkDescriptorUpdateTemplate    *desc_template;
    int                            desc_layout_num;
    int                            descriptor_sets_num;
    int                            total_descriptor_sets;
    int                            pool_size_desc_num;

    /* Temporary, used to store data in between initialization stages */
    VkDescriptorUpdateTemplateCreateInfo *desc_template_info;
    VkDescriptorPoolSize *pool_size_desc;
} FFVulkanPipeline;

typedef struct FFVkQueueCtx {
    VkFence fence;
    VkQueue queue;

    /* Buffer dependencies */
    AVBufferRef **buf_deps;
    int nb_buf_deps;
    int buf_deps_alloc_size;

    /* Frame dependencies */
    AVFrame **frame_deps;
    int nb_frame_deps;
    int frame_deps_alloc_size;
} FFVkQueueCtx;

typedef struct FFVkExecContext {
    FFVkQueueFamilyCtx *qf;

    VkCommandPool pool;
    VkCommandBuffer *bufs;
    FFVkQueueCtx *queues;

    AVBufferRef ***deps;
    int *nb_deps;
    int *dep_alloc_size;

    FFVulkanPipeline *bound_pl;

    VkSemaphore *sem_wait;
    int sem_wait_alloc; /* Allocated sem_wait */
    int sem_wait_cnt;

    uint64_t *sem_wait_val;
    int sem_wait_val_alloc;

    VkPipelineStageFlagBits *sem_wait_dst;
    int sem_wait_dst_alloc; /* Allocated sem_wait_dst */

    VkSemaphore *sem_sig;
    int sem_sig_alloc; /* Allocated sem_sig */
    int sem_sig_cnt;

    uint64_t *sem_sig_val;
    int sem_sig_val_alloc;

    uint64_t **sem_sig_val_dst;
    int sem_sig_val_dst_alloc;
} FFVkExecContext;

typedef struct FFVulkanContext {
    const AVClass         *class;
    FFVulkanFunctions     vkfn;
    FFVulkanExtensions    extensions;
    VkPhysicalDeviceProperties props;
    VkPhysicalDeviceMemoryProperties mprops;

    AVBufferRef           *device_ref;
    AVBufferRef           *frames_ref; /* For in-place filtering */
    AVHWDeviceContext     *device;
    AVVulkanDeviceContext *hwctx;

    /* Properties */
    int                 output_width;
    int                output_height;
    enum AVPixelFormat output_format;
    enum AVPixelFormat  input_format;

    /* Samplers */
    FFVkSampler **samplers;
    int samplers_num;

    /* Exec contexts */
    FFVkExecContext **exec_ctx;
    int exec_ctx_num;

    /* Pipelines (each can have 1 shader of each type) */
    FFVulkanPipeline **pipelines;
    int pipelines_num;

    void *scratch; /* Scratch memory used only in functions */
    unsigned int scratch_size;
} FFVulkanContext;

/* Identity mapping - r = r, b = b, g = g, a = a */
extern const VkComponentMapping ff_comp_identity_map;

/**
 * General lavfi IO functions
 */
int  ff_vk_filter_init                 (AVFilterContext *avctx);
int  ff_vk_filter_config_input         (AVFilterLink   *inlink);
int  ff_vk_filter_config_output        (AVFilterLink  *outlink);
int  ff_vk_filter_config_output_inplace(AVFilterLink  *outlink);
void ff_vk_filter_uninit               (AVFilterContext *avctx);

/**
 * Converts Vulkan return values to strings
 */
const char *ff_vk_ret2str(VkResult res);

/**
 * Returns 1 if the image is any sort of supported RGB
 */
int ff_vk_mt_is_np_rgb(enum AVPixelFormat pix_fmt);

/**
 * Gets the glsl format string for a pixel format
 */
const char *ff_vk_shader_rep_fmt(enum AVPixelFormat pixfmt);

/**
 * Initialize a queue family.
 * A queue limit of 0 means no limit.
 */
void ff_vk_qf_init(AVFilterContext *avctx, FFVkQueueFamilyCtx *qf,
                   VkQueueFlagBits dev_family, int queue_limit);

/**
 * Rotate through the queues in a queue family.
 */
void ff_vk_qf_rotate(FFVkQueueFamilyCtx *qf);

/**
 * Create a Vulkan sampler, will be auto-freed in ff_vk_filter_uninit()
 */
FFVkSampler *ff_vk_init_sampler(AVFilterContext *avctx, int unnorm_coords,
                                VkFilter filt);

/**
 * Create an imageview.
 * Guaranteed to remain alive until the queue submission has finished executing,
 * and will be destroyed after that.
 */
int ff_vk_create_imageview(AVFilterContext *avctx, FFVkExecContext *e,
                           VkImageView *v, VkImage img, VkFormat fmt,
                           const VkComponentMapping map);

/**
 * Define a push constant for a given stage into a pipeline.
 * Must be called before the pipeline layout has been initialized.
 */
int ff_vk_add_push_constant(AVFilterContext *avctx, FFVulkanPipeline *pl,
                            int offset, int size, VkShaderStageFlagBits stage);

/**
 * Inits a pipeline. Everything in it will be auto-freed when calling
 * ff_vk_filter_uninit().
 */
FFVulkanPipeline *ff_vk_create_pipeline(AVFilterContext *avctx,
                                      FFVkQueueFamilyCtx *qf);

/**
 * Inits a shader for a specific pipeline. Will be auto-freed on uninit.
 */
FFSPIRVShader *ff_vk_init_shader(AVFilterContext *avctx, FFVulkanPipeline *pl,
                                 const char *name, VkShaderStageFlags stage);

/**
 * Writes the workgroup size for a shader.
 */
void ff_vk_set_compute_shader_sizes(AVFilterContext *avctx, FFSPIRVShader *shd,
                                    int local_size[3]);

/**
 * Adds a descriptor set to the shader and registers them in the pipeline.
 */
int ff_vk_add_descriptor_set(AVFilterContext *avctx, FFVulkanPipeline *pl,
                             FFSPIRVShader *shd, FFVulkanDescriptorSetBinding *desc,
                             int num, int only_print_to_shader);

/**
 * Compiles the shader, entrypoint must be set to "main".
 */
int ff_vk_compile_shader(AVFilterContext *avctx, FFSPIRVShader *shd,
                         const char *entrypoint);

/**
 * Pretty print shader, mainly used by shader compilers.
 */
void ff_vk_print_shader(AVFilterContext *avctx, FFSPIRVShader *shd, int prio);

/**
 * Initializes the pipeline layout after all shaders and descriptor sets have
 * been finished.
 */
int ff_vk_init_pipeline_layout(AVFilterContext *avctx, FFVulkanPipeline *pl);

/**
 * Initializes a compute pipeline. Will pick the first shader with the
 * COMPUTE flag set.
 */
int ff_vk_init_compute_pipeline(AVFilterContext *avctx, FFVulkanPipeline *pl);

/**
 * Updates a descriptor set via the updaters defined.
 * Can be called immediately after pipeline creation, but must be called
 * at least once before queue submission.
 */
void ff_vk_update_descriptor_set(AVFilterContext *avctx, FFVulkanPipeline *pl,
                                 int set_id);

/**
 * Init an execution context for command recording and queue submission.
 * WIll be auto-freed on uninit.
 */
int ff_vk_create_exec_ctx(AVFilterContext *avctx, FFVkExecContext **ctx,
                          FFVkQueueFamilyCtx *qf);

/**
 * Begin recording to the command buffer. Previous execution must have been
 * completed, which ff_vk_submit_exec_queue() will ensure.
 */
int ff_vk_start_exec_recording(AVFilterContext *avctx, FFVkExecContext *e);

/**
 * Add a command to bind the completed pipeline and its descriptor sets.
 * Must be called after ff_vk_start_exec_recording() and before submission.
 */
void ff_vk_bind_pipeline_exec(AVFilterContext *avctx, FFVkExecContext *e,
                              FFVulkanPipeline *pl);

/**
 * Updates push constants.
 * Must be called after binding a pipeline if any push constants were defined.
 */
void ff_vk_update_push_exec(AVFilterContext *avctx, FFVkExecContext *e,
                            VkShaderStageFlagBits stage, int offset,
                            size_t size, void *src);

/**
 * Gets the command buffer to use for this submission from the exe context.
 */
VkCommandBuffer ff_vk_get_exec_buf(AVFilterContext *avctx, FFVkExecContext *e);

/**
 * Adds a generic AVBufferRef as a queue depenency.
 */
int ff_vk_add_dep_exec_ctx(AVFilterContext *avctx, FFVkExecContext *e,
                           AVBufferRef **deps, int nb_deps);

/**
 * Discards all queue dependencies
 */
void ff_vk_discard_exec_deps(AVFilterContext *avctx, FFVkExecContext *e);

/**
 * Adds a frame as a queue dependency. This also manages semaphore signalling.
 * Must be called before submission.
 */
int ff_vk_add_exec_dep(AVFilterContext *avctx, FFVkExecContext *e,
                       AVFrame *frame, VkPipelineStageFlagBits in_wait_dst_flag);

/**
 * Submits a command buffer to the queue for execution.
 * Will block until execution has finished in order to simplify resource
 * management.
 */
int ff_vk_submit_exec_queue(AVFilterContext *avctx, FFVkExecContext *e);

/**
 * Create a VkBuffer with the specified parameters.
 */
int ff_vk_create_buf(AVFilterContext *avctx, FFVkBuffer *buf, size_t size,
                     VkBufferUsageFlags usage, VkMemoryPropertyFlagBits flags);

/**
 * Maps the buffer to userspace. Set invalidate to 1 if reading the contents
 * is necessary.
 */
int ff_vk_map_buffers(AVFilterContext *avctx, FFVkBuffer *buf, uint8_t *mem[],
                      int nb_buffers, int invalidate);

/**
 * Unmaps the buffer from userspace. Set flush to 1 to write and sync.
 */
int ff_vk_unmap_buffers(AVFilterContext *avctx, FFVkBuffer *buf, int nb_buffers,
                        int flush);

/**
 * Frees a buffer.
 */
void ff_vk_free_buf(AVFilterContext *avctx, FFVkBuffer *buf);

#endif /* AVFILTER_VULKAN_H */

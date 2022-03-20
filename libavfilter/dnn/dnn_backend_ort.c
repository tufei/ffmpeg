/*
 * Copyright (c) 2020 Ngyu-Phee Yen
 *
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

/**
 * @file
 * DNN onnxruntime backend implementation.
 */

#include "dnn_backend_ort.h"
#include "dnn_backend_native.h"
#include "libavformat/avio.h"
#include "libavutil/avassert.h"
#include "libavutil/avstring.h"
#include "libavutil/cpu.h"
#include "libavcodec/defs.h"
#include "../internal.h"
#include "dnn_io_proc.h"
#include "dnn_backend_common.h"
#include "safe_queue.h"
#include <onnxruntime_c_api.h>

typedef struct ORTOptions {
    uint8_t async;
    uint32_t nireq;
    char *opt_model_filename;
    char *profile_file_prefix;
    char *logger_id;
    int intra_op_threads;
    int inter_op_threads;
    int execute_mode;
    int opt_level;
    int log_verbosity_level;
    int log_severity_level;
    int enable_cuda;
} ORTOptions;

typedef struct ORTContext {
    const AVClass *class;
    ORTOptions options;
    OrtCUDAProviderOptions cuda_options;
} ORTContext;

typedef struct ORTModel {
    ORTContext ctx;
    DNNModel *model;
    const OrtApi *ort;
    OrtAllocator *allocator;
    OrtEnv *env;
    OrtSessionOptions *options;
    OrtSession *session;
    OrtStatus *status;
    SafeQueue *request_queue;
    Queue *lltask_queue;
    Queue *task_queue;
} ORTModel;

/**
 * Stores execution parameters for single
 * call to the ONNXRunTime C API
 */
typedef struct ORTInferRequest {
    const OrtApi *ort;
    OrtValue *input_tensor;
    OrtValue *output_tensor;
} ORTInferRequest;

typedef struct ORTRequestItem {
    ORTInferRequest *infer_request;
    LastLevelTaskItem *lltask;
    OrtStatus *status;
    DNNAsyncExecModule exec_module;
} ORTRequestItem;

#define OFFSET(x) offsetof(ORTContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM
static const AVOption dnn_onnxruntime_options[] = {
    { "opt_model_filename",
      "file name for optimized model after graph-level transformation",
      OFFSET(options.opt_model_filename),
      AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, FLAGS },
    { "profile_file_prefix",
      "file prefix for profiling result",
      OFFSET(options.profile_file_prefix),
      AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, FLAGS },
    { "logger_id",
      "logger ID for session output",
      OFFSET(options.logger_id),
      AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, FLAGS },
    { "intra_op_threads",
      "number of threads within nodes",
      OFFSET(options.intra_op_threads),
      AV_OPT_TYPE_INT, { .i64 = 0 }, 0, INT_MAX, FLAGS },
    { "inter_op_threads",
      "number of threads across nodes",
      OFFSET(options.inter_op_threads),
      AV_OPT_TYPE_INT, { .i64 = 0 }, 0, INT_MAX, FLAGS },
    { "execution_mode",
      "execute operators in sequential or parallel mode",
      OFFSET(options.execute_mode),
      AV_OPT_TYPE_INT, { .i64 = ORT_PARALLEL },
      ORT_SEQUENTIAL, ORT_PARALLEL, FLAGS },
    { "opt_level",
      "graph optimization level",
      OFFSET(options.opt_level),
      AV_OPT_TYPE_INT, { .i64 = ORT_ENABLE_ALL },
      ORT_DISABLE_ALL, ORT_ENABLE_ALL, FLAGS },
    { "log_verbosity_level",
      "logging verbosity level",
      OFFSET(options.log_verbosity_level),
      AV_OPT_TYPE_INT, { .i64 = ORT_LOGGING_LEVEL_WARNING },
      ORT_LOGGING_LEVEL_VERBOSE, ORT_LOGGING_LEVEL_FATAL, FLAGS },
    { "log_severity_level",
      "logging severity level",
      OFFSET(options.log_severity_level),
      AV_OPT_TYPE_INT, { .i64 = ORT_LOGGING_LEVEL_ERROR },
      ORT_LOGGING_LEVEL_VERBOSE, ORT_LOGGING_LEVEL_FATAL, FLAGS },
    { "enable_cuda",
      "enable CUDA execution provider when using ONNXRunTime_GPU",
      OFFSET(options.enable_cuda),
      AV_OPT_TYPE_INT, { .i64 = 1 }, 0, 1, FLAGS },
    DNN_BACKEND_COMMON_OPTIONS
    { NULL }
};

AVFILTER_DEFINE_CLASS(dnn_onnxruntime);

static int execute_model_ort(ORTRequestItem *request, Queue *lltask_queue);
static void infer_completion_callback(void *args);
static inline void destroy_request_item(ORTRequestItem **arg);

static void release_ort_context(ORTModel *ort_model)
{
    if (ort_model->session) {
        ort_model->ort->ReleaseSession(ort_model->session);
    }
    if (ort_model->options) {
        ort_model->ort->ReleaseSessionOptions(ort_model->options);
    }
    if (ort_model->status) {
        ort_model->ort->ReleaseStatus(ort_model->status);
    }
    if (ort_model->env) {
        ort_model->ort->ReleaseEnv(ort_model->env);
    }
}

static void ort_free_request(ORTInferRequest *request)
{
    if (!request)
        return;
    if (request->input_tensor) {
        request->ort->ReleaseValue(request->input_tensor);
        request->input_tensor = NULL;
    }
    if (request->output_tensor) {
        request->ort->ReleaseValue(request->output_tensor);
        request->output_tensor = NULL;
    }
}

static ORTInferRequest *ort_create_inference_request(const OrtApi *ort)
{
    ORTInferRequest *infer_request = av_malloc(sizeof(ORTInferRequest));
    if (!infer_request) {
        return NULL;
    }
    infer_request->ort = ort;
    infer_request->input_tensor = NULL;
    infer_request->output_tensor = NULL;
    return infer_request;
}

static int ort_start_inference(void *args)
{
    ORTRequestItem *request = args;
    ORTInferRequest *infer_request = request->infer_request;
    LastLevelTaskItem *lltask = request->lltask;
    TaskItem *task = lltask->task;
    ORTModel *ort_model = task->model;
    const OrtApi *ort = ort_model->ort;
    OrtStatus **status = &request->status;

    if (!request) {
        av_log(&ort_model->ctx, AV_LOG_ERROR, "ORTRequestItem is NULL\n");
        return DNN_GENERIC_ERROR;
    }

    *status = ort->Run(ort_model->session, NULL, &task->input_name,
                       (const OrtValue * const *)&infer_request->input_tensor,
                       1, task->output_names, 1, &infer_request->output_tensor);
    if (*status) {
        av_log(&ort_model->ctx, AV_LOG_ERROR, "Run(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        ort_free_request(infer_request);
        if (ff_safe_queue_push_back(ort_model->request_queue, request) < 0) {
            destroy_request_item(&request);
        }
        return DNN_GENERIC_ERROR;
    }

    return 0;
}

static inline void destroy_request_item(ORTRequestItem **arg) {
    ORTRequestItem *request;
    if (!arg) {
        return;
    }
    request = *arg;
    ort_free_request(request->infer_request);
    av_freep(&request->infer_request);
    av_freep(&request->lltask);
    av_freep(arg);
}

static int extract_lltask_from_task(TaskItem *task, Queue *lltask_queue)
{
    ORTModel *ort_model = task->model;
    ORTContext *ctx = &ort_model->ctx;
    LastLevelTaskItem *lltask = av_malloc(sizeof(*lltask));
    if (!lltask) {
        av_log(ctx, AV_LOG_ERROR,
               "Unable to allocate space for LastLevelTaskItem\n");
        return DNN_GENERIC_ERROR;
    }
    task->inference_todo = 1;
    task->inference_done = 0;
    lltask->task = task;
    if (ff_queue_push_back(lltask_queue, lltask) < 0) {
        av_log(ctx, AV_LOG_ERROR, "Failed to push back lltask_queue.\n");
        av_freep(&lltask);
        return DNN_GENERIC_ERROR;
    }
    return 0;
}

static int allocate_input_tensor(ORTModel *ort_model, const DNNData *input,
                                 OrtValue **out)
{
    ORTContext *ctx = &ort_model->ctx;
    const OrtApi *ort = ort_model->ort;
    OrtStatus **status = &ort_model->status;
    ONNXTensorElementDataType dt;
    int64_t input_dims[] = {1, input->channels, input->height, input->width};
    int is_tensor = 0;

    switch (input->dt) {
    case DNN_FLOAT:
        dt = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        break;
    case DNN_UINT8:
        dt = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        break;
    default:
        av_assert0(!"should not reach here");
    }

    *status = ort->CreateTensorAsOrtValue(ort_model->allocator, input_dims,
                                          sizeof(input_dims) / sizeof(int64_t),
                                          dt, out);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "CreateTensorAsOrtValue(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_GENERIC_ERROR;
    }

    *status = ort->IsTensor(*out, &is_tensor);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "IsTensor(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_GENERIC_ERROR;
    }
    av_assert0(1 == is_tensor);

    return 0;
}

static int get_output_tensor(void *model, DNNData * const output,
                             OrtValue *output_tensor)
{
    ORTModel *ort_model = (ORTModel *)model;
    ORTContext *ctx = &ort_model->ctx;
    const OrtApi *ort = ort_model->ort;
    OrtStatus **status = &ort_model->status;
    OrtTensorTypeAndShapeInfo *tensor = NULL;
    ONNXTensorElementDataType data_type;
    int64_t dims[4];
    size_t num_dims = 0;

    *status = ort->GetTensorTypeAndShape(output_tensor, &tensor);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "GetTensorTypeAndShape(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_GENERIC_ERROR;
    }

    *status = ort->GetTensorElementType(tensor, &data_type);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "GetTensorElementType(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_GENERIC_ERROR;
    }
    switch (data_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        output->dt = DNN_FLOAT;
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        output->dt = DNN_UINT8;
        break;
    default:
        av_log(ctx, AV_LOG_ERROR, "Unsupported tensor element data type\n");
        return DNN_GENERIC_ERROR;
    }

    *status = ort->GetDimensionsCount(tensor, &num_dims);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "GetDimensionsCount(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_GENERIC_ERROR;
    }
    av_assert0(4 == num_dims);

    *status = ort->GetDimensions(tensor, dims, num_dims);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "GetDimensionsCount(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_GENERIC_ERROR;
    }

    ort->ReleaseTensorTypeAndShapeInfo(tensor);

    av_assert0(dims[0] == 1);
    output->channels = dims[1];
    output->height = dims[2];
    output->width = dims[3];

    *status = ort->GetTensorMutableData(output_tensor, &output->data);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "GetTensorMutableData(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_GENERIC_ERROR;
    }

    return 0;
}

static int get_input_ort(void *model, DNNData *input, const char *input_name)
{
    ORTModel *ort_model = (ORTModel *)model;
    ORTContext *ctx = &ort_model->ctx;
    const OrtApi *ort = ort_model->ort;
    OrtStatus **status = &ort_model->status;
    OrtTypeInfo *type = NULL;
    const OrtTensorTypeAndShapeInfo *tensor = NULL;
    ONNXTensorElementDataType data_type;
    int64_t dims[4];
    size_t index = -1;
    size_t num_inputs = 0;
    size_t num_dims = 0;

    *status = ort->SessionGetInputCount(ort_model->session, &num_inputs);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "SessionGetInputCount(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_GENERIC_ERROR;
    }
    for (size_t i = 0; i < num_inputs; ++i) {
        char *index_name = NULL;
        int str_cmp = -1;
        *status = ort->SessionGetInputName(ort_model->session, i,
                                           ort_model->allocator, &index_name);
        if (*status) {
            av_log(ctx, AV_LOG_ERROR, "SessionGetInputName(): %s\n",
                   ort->GetErrorMessage(*status));
            ort->ReleaseStatus(*status);
            return DNN_GENERIC_ERROR;
        }
        str_cmp = strncmp(input_name, index_name, 1024);
        *status = ort->AllocatorFree(ort_model->allocator, index_name);
        if (*status) {
            av_log(ctx, AV_LOG_ERROR, "AllocatorFree(): %s\n",
                   ort->GetErrorMessage(*status));
            ort->ReleaseStatus(*status);
            return DNN_GENERIC_ERROR;
        }
        if (0 == str_cmp) {
            index = i;
            break;
        }
    }
    if (-1 == index) {
        av_log(ctx, AV_LOG_ERROR, "Could not find \"%s\" in model\n", input_name);
        return DNN_GENERIC_ERROR;
    }

    *status = ort->SessionGetInputTypeInfo(ort_model->session, index, &type);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "SessionGetInputTypeInfo(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_GENERIC_ERROR;
    }
    *status = ort->CastTypeInfoToTensorInfo(type, &tensor);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "CastTypeInfoToTensorInfo(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_GENERIC_ERROR;
    }

    *status = ort->GetTensorElementType(tensor, &data_type);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "GetTensorElementType(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_GENERIC_ERROR;
    }
    switch (data_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        input->dt = DNN_FLOAT;
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        input->dt = DNN_UINT8;
        break;
    default:
        av_log(ctx, AV_LOG_ERROR, "Unsupported tensor element data type\n");
        return DNN_GENERIC_ERROR;
    }

    *status = ort->GetDimensionsCount(tensor, &num_dims);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "GetDimensionsCount(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_GENERIC_ERROR;
    }
    av_assert0(4 == num_dims);

    *status = ort->GetDimensions(tensor, dims, num_dims);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "GetDimensionsCount(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_GENERIC_ERROR;
    }

    ort->ReleaseTypeInfo(type);

    av_assert0(dims[0] == 1);
    input->channels = dims[1];
    input->height = dims[2];
    input->width = dims[3];

    return 0;
}

static int get_output_ort(void *model, const char *input_name,
                          int input_width, int input_height,
                          const char *output_name,
                          int *output_width, int *output_height)
{
    int ret;
    ORTModel *ort_model = (ORTModel *)model;
    ORTContext *ctx = &ort_model->ctx;
    TaskItem task;
    ORTRequestItem *request;
    DNNExecBaseParams exec_params = {
        .input_name     = input_name,
        .output_names   = &output_name,
        .nb_output      = 1,
        .in_frame       = NULL,
        .out_frame      = NULL,
    };

    if (ff_dnn_fill_gettingoutput_task(&task, &exec_params, ort_model,
                                       input_height, input_width, ctx) != 0) {
        goto err;
    }

    if (extract_lltask_from_task(&task, ort_model->lltask_queue) != 0) {
        av_log(ctx, AV_LOG_ERROR, "unable to extract inference from task.\n");
        ret = DNN_GENERIC_ERROR;
        goto err;
    }

    request = ff_safe_queue_pop_front(ort_model->request_queue);
    if (!request) {
        av_log(ctx, AV_LOG_ERROR, "unable to get infer request.\n");
        ret = DNN_GENERIC_ERROR;
        goto err;
    }

    ret = execute_model_ort(request, ort_model->lltask_queue);
    *output_width = task.out_frame->width;
    *output_height = task.out_frame->height;

err:
    av_frame_free(&task.out_frame);
    av_frame_free(&task.in_frame);
    return ret;
}

static int load_ort_model(ORTModel *ort_model, const char *model_filename)
{
    ORTContext *ctx = &ort_model->ctx;
    const OrtApi *ort = NULL;
    OrtStatus **status = &ort_model->status;
    const OrtApiBase *oab = NULL;

    oab = OrtGetApiBase();
    if (NULL == oab) {
        av_log(ctx, AV_LOG_ERROR, "Error calling OrtGetApiBase()\n");
        return DNN_GENERIC_ERROR;
    }

    ort_model->ort = oab->GetApi(ORT_API_VERSION);
    if (NULL == ort_model->ort) {
        av_log(ctx, AV_LOG_ERROR, "Error calling GetApi()\n");
        return DNN_GENERIC_ERROR;
    }
    ort = ort_model->ort;

    *status = ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ort_inference",
                             &ort_model->env);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "CreateEnv(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_GENERIC_ERROR;
    }

    *status = ort->CreateSessionOptions(&ort_model->options);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "CreateSessionOptions(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_GENERIC_ERROR;
    }

    *status = ort->SetSessionExecutionMode(ort_model->options,
                                           ctx->options.execute_mode);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "SetSessionExecutionMode(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_GENERIC_ERROR;
    }

    if (ORT_SEQUENTIAL == ctx->options.execute_mode) {
        *status = ort->EnableMemPattern(ort_model->options);
        if (*status) {
            av_log(ctx, AV_LOG_ERROR, "EnableMemPattern(): %s\n",
                   ort->GetErrorMessage(*status));
            ort->ReleaseStatus(*status);
            return DNN_GENERIC_ERROR;
        }
    }

    if (ctx->options.profile_file_prefix) {
        *status = ort->EnableProfiling(ort_model->options,
                                       ctx->options.profile_file_prefix);
        if (*status) {
            av_log(ctx, AV_LOG_ERROR, "EnableProfiling(): %s\n",
                   ort->GetErrorMessage(*status));
            ort->ReleaseStatus(*status);
            return DNN_GENERIC_ERROR;
        }
    } else {
        *status = ort->DisableProfiling(ort_model->options);
        if (*status) {
            av_log(ctx, AV_LOG_ERROR, "DisableProfiling(): %s\n",
                   ort->GetErrorMessage(*status));
            ort->ReleaseStatus(*status);
            return DNN_GENERIC_ERROR;
        }
    }

    if (ctx->options.logger_id) {
        *status = ort->SetSessionLogId(ort_model->options,
                                       ctx->options.logger_id);
        if (*status) {
            av_log(ctx, AV_LOG_ERROR, "SetSessionLogId(): %s\n",
                   ort->GetErrorMessage(*status));
            ort->ReleaseStatus(*status);
            return DNN_GENERIC_ERROR;
        }
    }

    if (ctx->options.enable_cuda) {
        memset(&ctx->cuda_options, 0, sizeof(OrtCUDAProviderOptions));
        ctx->cuda_options.cudnn_conv_algo_search =
            OrtCudnnConvAlgoSearchDefault;
        ctx->cuda_options.gpu_mem_limit = -1UL;

        *status =
            ort->SessionOptionsAppendExecutionProvider_CUDA(ort_model->options,
                                                            &ctx->cuda_options);
        if (*status) {
            av_log(ctx, AV_LOG_ERROR, "SessionOptionsAppendExecutionProvider_CUDA(): %s\n",
                   ort->GetErrorMessage(*status));
            ort->ReleaseStatus(*status);
            return DNN_GENERIC_ERROR;
        }
    }

    *status = ort->SetSessionLogVerbosityLevel(ort_model->options,
                                               ctx->options.log_verbosity_level);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "SetSessionLogVerbosityLevel(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_GENERIC_ERROR;
    }

    *status = ort->SetSessionLogSeverityLevel(ort_model->options,
                                              ctx->options.log_severity_level);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "SetSessionLogSeverityLevel(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_GENERIC_ERROR;
    }

    *status = ort->SetSessionGraphOptimizationLevel(ort_model->options,
                                                    ctx->options.opt_level);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "SetSessionGraphOptimizationLevel(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_GENERIC_ERROR;
    }

    *status = ort->SetIntraOpNumThreads(ort_model->options,
                                        ctx->options.intra_op_threads);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "SetIntraOpNumThreads(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_GENERIC_ERROR;
    }

    *status = ort->SetInterOpNumThreads(ort_model->options,
                                        ctx->options.inter_op_threads);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "SetInterOpNumThreads(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_GENERIC_ERROR;
    }

    *status = ort->EnableCpuMemArena(ort_model->options);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "EnableCpuMemArena(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_GENERIC_ERROR;
    }

    *status = ort->GetAllocatorWithDefaultOptions(&ort_model->allocator);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "GetAllocatorWithDefaultOptions(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_GENERIC_ERROR;
    }

    *status = ort->CreateSession(ort_model->env, model_filename,
                                 ort_model->options, &ort_model->session);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "CreateSession(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_GENERIC_ERROR;
    }

    return 0;
}

DNNModel *ff_dnn_load_model_ort(const char *model_filename,
                                DNNFunctionType func_type,
                                const char *options, AVFilterContext *filter_ctx)
{
    DNNModel *model = NULL;
    ORTModel *ort_model = NULL;
    ORTContext *ctx = NULL;

    model = av_mallocz(sizeof(DNNModel));
    if (!model) {
        return NULL;
    }

    ort_model = av_mallocz(sizeof(ORTModel));
    if (!ort_model) {
        av_freep(&model);
        return NULL;
    }
    ort_model->model = model;
    ctx = &ort_model->ctx;
    ctx->class = &dnn_onnxruntime_class;

    //parse options
    av_opt_set_defaults(ctx);
    if (av_opt_set_from_string(ctx, options, NULL, "=", "&") < 0) {
        av_log(&ort_model->ctx, AV_LOG_ERROR,
               "Failed to parse options \"%s\"\n", options);
        goto err;
    }

    if (load_ort_model(ort_model, model_filename) != 0) {
        goto err;
    }

    if (ctx->options.nireq <= 0) {
        ctx->options.nireq = av_cpu_count() / 2 + 1;
    }

    ctx->options.async = 0;
#if !HAVE_PTHREAD_CANCEL
    if (ctx->options.async) {
        ctx->options.async = 0;
        av_log(filter_ctx, AV_LOG_WARNING,
               "pthread is not supported, roll back to sync.\n");
    }
#endif

    ort_model->request_queue = ff_safe_queue_create();
    if (!ort_model->request_queue) {
        goto err;
    }

    for (int i = 0; i < ctx->options.nireq; i++) {
        ORTRequestItem *item = av_mallocz(sizeof(*item));
        if (!item) {
            goto err;
        }
        item->lltask = NULL;
        item->infer_request = ort_create_inference_request(ort_model->ort);
        if (!item->infer_request) {
            av_log(ctx, AV_LOG_ERROR,
                   "Failed to allocate memory for ONNXRunTime inference request\n");
            av_freep(&item);
            goto err;
        }
        item->status = NULL;
        item->exec_module.start_inference = &ort_start_inference;
        item->exec_module.callback = &infer_completion_callback;
        item->exec_module.args = item;

        if (ff_safe_queue_push_back(ort_model->request_queue, item) < 0) {
            destroy_request_item(&item);
            goto err;
        }
    }

    ort_model->lltask_queue = ff_queue_create();
    if (!ort_model->lltask_queue) {
        goto err;
    }

    ort_model->task_queue = ff_queue_create();
    if (!ort_model->task_queue) {
        goto err;
    }

    model->model = (void *)ort_model;
    model->get_input = &get_input_ort;
    model->get_output = &get_output_ort;
    model->options = options;
    model->filter_ctx = filter_ctx;
    model->func_type = func_type;

    return model;
err:
    ff_dnn_free_model_ort(&model);
    return NULL;
}

static int fill_model_input_ort(ORTModel *ort_model, ORTRequestItem *request)
{
    DNNData input;
    LastLevelTaskItem *lltask;
    TaskItem *task;
    ORTInferRequest *infer_request;
    ORTContext *ctx = &ort_model->ctx;
    const OrtApi *ort = ort_model->ort;
    size_t num_outputs = 0;
    OrtStatus **status = &request->status;

    lltask = ff_queue_pop_front(ort_model->lltask_queue);
    av_assert0(lltask);
    task = lltask->task;
    request->lltask = lltask;

    if (get_input_ort(ort_model, &input, task->input_name) != 0) {
        goto err;
    }

    infer_request = request->infer_request;
    input.height = task->in_frame->height;
    input.width = task->in_frame->width;

    if (0 != allocate_input_tensor(ort_model, &input,
                                             &infer_request->input_tensor)) {
        av_log(ctx, AV_LOG_ERROR, "Cannot allocate memory for input tensor\n");
        goto err;
    }

    *status = ort->GetTensorMutableData(infer_request->input_tensor,
                                        &input.data);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "GetTensorMutableData(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        goto err;
    }

    switch (ort_model->model->func_type) {
    case DFT_PROCESS_FRAME:
        if (task->do_ioproc) {
            if (ort_model->model->frame_pre_proc != NULL) {
                ort_model->model->frame_pre_proc(task->in_frame, &input,
                                                 ort_model->model->filter_ctx);
            } else {
                ff_proc_from_frame_to_dnn(task->in_frame, &input, ctx);
            }
        }
        break;
    case DFT_ANALYTICS_DETECT:
        ff_frame_to_dnn_detect(task->in_frame, &input, ctx);
        break;
    default:
        avpriv_report_missing_feature(ctx, "model function type %d",
                                      ort_model->model->func_type);
        break;
    }

    if (task->nb_output != 1) {
        // currently, the filter does not need multiple outputs,
        // so we just pending the support until we really need it.
        av_log(ctx, AV_LOG_ERROR, "Do not support multiple outputs\n");
        goto err;
    }

    *status = ort->SessionGetOutputCount(ort_model->session, &num_outputs);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "SessionGetOutputCount(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        goto err;
    }

    if (num_outputs != 1) {
        // currently, follow the given outputs
        av_log(ctx, AV_LOG_ERROR, "Model has multiple outputs\n");
        goto err;
    }

    return 0;
err:
    ort_free_request(infer_request);
    return DNN_GENERIC_ERROR;
}

static void infer_completion_callback(void *args) {
    ORTRequestItem *request = args;
    LastLevelTaskItem *lltask = request->lltask;
    TaskItem *task = lltask->task;
    DNNData output;
    int is_tensor = 0;
    ORTInferRequest *infer_request = request->infer_request;
    ORTModel *ort_model = task->model;
    ORTContext *ctx = &ort_model->ctx;
    const OrtApi *ort = ort_model->ort;
    OrtStatus **status = &request->status;

    *status = ort->IsTensor(infer_request->output_tensor, &is_tensor);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "IsTensor(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        goto err;
    }
    av_assert0(1 == is_tensor);

    if (0 != get_output_tensor(ort_model, &output,
                                         infer_request->output_tensor)) {
        av_log(ctx, AV_LOG_ERROR, "Cannot get output\n");
        goto err;
    }

    switch (ort_model->model->func_type) {
    case DFT_PROCESS_FRAME:
        //it only support 1 output if it's frame in & frame out
        if (task->do_ioproc) {
            if (ort_model->model->frame_post_proc != NULL) {
                ort_model->model->frame_post_proc(task->out_frame, &output,
                                                  ort_model->model->filter_ctx);
            } else {
                ff_proc_from_dnn_to_frame(task->out_frame, &output, ctx);
            }
        } else {
            task->out_frame->width = output.width;
            task->out_frame->height = output.height;
        }
        break;
    case DFT_ANALYTICS_DETECT:
        if (!ort_model->model->detect_post_proc) {
            av_log(ctx, AV_LOG_ERROR, "Detect filter needs provide post proc\n");
            return;
        }
        ort_model->model->detect_post_proc(task->in_frame, &output,
                                           task->nb_output,
                                           ort_model->model->filter_ctx);
        break;
    default:
        av_log(ctx, AV_LOG_ERROR,
               "ONNXRunTime backend does not support this kind of dnn filter now\n");
        goto err;
    }
    task->inference_done++;
err:
    ort_free_request(infer_request);

    if (ff_safe_queue_push_back(ort_model->request_queue, request) < 0) {
        destroy_request_item(&request);
        av_log(ctx, AV_LOG_ERROR, "Failed to push back request_queue.\n");
    }
}

static int execute_model_ort(ORTRequestItem *request, Queue *lltask_queue)
{
    ORTModel *ort_model;
    ORTContext *ctx;
    LastLevelTaskItem *lltask;
    TaskItem *task;

    if (ff_queue_size(lltask_queue) == 0) {
        destroy_request_item(&request);
        return 0;
    }

    lltask = ff_queue_peek_front(lltask_queue);
    task = lltask->task;
    ort_model = task->model;
    ctx = &ort_model->ctx;

    if (fill_model_input_ort(ort_model, request) != 0) {
        goto err;
    }

    if (task->async) {
        if (ff_dnn_start_inference_async(ctx, &request->exec_module) !=
            0) {
            goto err;
        }
        return 0;
    } else {
        if (ort_start_inference(request) != 0) {
            goto err;
        }
        infer_completion_callback(request);
        return task->inference_done == task->inference_todo ?
               0 : DNN_GENERIC_ERROR;
    }
err:
    ort_free_request(request->infer_request);
    if (ff_safe_queue_push_back(ort_model->request_queue, request) < 0) {
        destroy_request_item(&request);
    }
    return DNN_GENERIC_ERROR;
}

int ff_dnn_execute_model_ort(const DNNModel *model,
                             DNNExecBaseParams *exec_params)
{
    ORTModel *ort_model = (ORTModel *)model->model;
    ORTContext *ctx = &ort_model->ctx;
    TaskItem *task;
    ORTRequestItem *request;

    if (ff_check_exec_params(ctx, DNN_ORT, model->func_type, exec_params) != 0) {
         return DNN_GENERIC_ERROR;
    }

    task = av_malloc(sizeof(*task));
    if (!task) {
        av_log(ctx, AV_LOG_ERROR, "unable to alloc memory for task item.\n");
        return DNN_GENERIC_ERROR;
    }

    if (ff_dnn_fill_task(task, exec_params, ort_model, ctx->options.async, 1) !=
        0) {
        av_freep(&task);
        return DNN_GENERIC_ERROR;
    }

    if (ff_queue_push_back(ort_model->task_queue, task) < 0) {
        av_freep(&task);
        av_log(ctx, AV_LOG_ERROR, "unable to push back task_queue.\n");
        return DNN_GENERIC_ERROR;
    }

    if (extract_lltask_from_task(task, ort_model->lltask_queue) != 0) {
        av_log(ctx, AV_LOG_ERROR, "unable to extract last level task from task.\n");
        return DNN_GENERIC_ERROR;
    }

    request = ff_safe_queue_pop_front(ort_model->request_queue);
    if (!request) {
        av_log(ctx, AV_LOG_ERROR, "unable to get infer request.\n");
        return DNN_GENERIC_ERROR;
    }
    return execute_model_ort(request, ort_model->lltask_queue);
}

DNNAsyncStatusType ff_dnn_get_result_ort(const DNNModel *model, AVFrame **in,
                                         AVFrame **out)
{
    ORTModel *ort_model = model->model;
    return ff_dnn_get_result_common(ort_model->task_queue, in, out);
}

int ff_dnn_flush_ort(const DNNModel *model)
{
    ORTModel *ort_model = model->model;
    ORTContext *ctx = &ort_model->ctx;
    ORTRequestItem *request;
    int ret;

    if (ff_queue_size(ort_model->lltask_queue) == 0) {
        // no pending task need to flush
        return 0;
    }

    request = ff_safe_queue_pop_front(ort_model->request_queue);
    if (!request) {
        av_log(ctx, AV_LOG_ERROR, "unable to get infer request.\n");
        return DNN_GENERIC_ERROR;
    }

    ret = fill_model_input_ort(ort_model, request);
    if (ret != 0) {
        av_log(ctx, AV_LOG_ERROR, "Failed to fill model input.\n");
        if (ff_safe_queue_push_back(ort_model->request_queue, request) < 0) {
            destroy_request_item(&request);
        }
        return ret;
    }

    return ff_dnn_start_inference_async(ctx, &request->exec_module);
}

void ff_dnn_free_model_ort(DNNModel **model)
{
    ORTModel *ort_model;

    if (*model) {
        ort_model = (ORTModel *)(*model)->model;

        while (ff_safe_queue_size(ort_model->request_queue) != 0) {
            ORTRequestItem *item =
                ff_safe_queue_pop_front(ort_model->request_queue);
            destroy_request_item(&item);
        }
        ff_safe_queue_destroy(ort_model->request_queue);

        while (ff_queue_size(ort_model->lltask_queue) != 0) {
            LastLevelTaskItem *item = ff_queue_pop_front(ort_model->lltask_queue);
            av_freep(&item);
        }
        ff_queue_destroy(ort_model->lltask_queue);

        while (ff_queue_size(ort_model->task_queue) != 0) {
            TaskItem *item = ff_queue_pop_front(ort_model->task_queue);
            av_frame_free(&item->in_frame);
            av_frame_free(&item->out_frame);
            av_freep(&item);
        }
        ff_queue_destroy(ort_model->task_queue);

        release_ort_context(ort_model);
        av_freep(&ort_model);
        av_freep(model);
    }
}


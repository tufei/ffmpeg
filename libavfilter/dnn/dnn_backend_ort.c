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
#include "dnn_backend_native_layer_conv2d.h"
#include "dnn_backend_native_layer_depth2space.h"
#include "libavformat/avio.h"
#include "libavutil/avassert.h"
#include "../internal.h"
#include "dnn_backend_native_layer_pad.h"
#include "dnn_backend_native_layer_maximum.h"
#include "dnn_io_proc.h"

#include <onnxruntime_c_api.h>

typedef struct ORTOptions {
    char *opt_model_filename;
    char *profile_file_prefix;
    char *logger_id;
    int intra_op_threads;
    int inter_op_threads;
    int execute_mode;
    int opt_level;
    int log_verbosity_level;
    int log_severity_level;
} ORTOptions;

typedef struct ORTContext {
    const AVClass *class;
    ORTOptions options;
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
} ORTModel;

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
      AV_OPT_TYPE_INT, { .i64 = 1 }, 0, INT_MAX, FLAGS },
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
      AV_OPT_TYPE_INT, { .i64 = ORT_ENABLE_BASIC },
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
    { NULL }
};

AVFILTER_DEFINE_CLASS(dnn_onnxruntime);

static DNNReturnType execute_model_ort(const DNNModel *model,
                                       const char *input_name,
                                       AVFrame *in_frame,
                                       const char **output_names,
                                       uint32_t nb_output,
                                       AVFrame *out_frame,
                                       int do_ioproc);

static void free_buffer(void *data, size_t length)
{
    av_freep(&data);
}

static DNNReturnType allocate_input_tensor(const DNNModel *model,
                                           const DNNData *input,
                                           OrtValue **out)
{
    ORTModel *ort_model = (ORTModel *)model;
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
        return DNN_ERROR;
    }

    *status = ort->IsTensor(*out, &is_tensor);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "IsTensor(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_ERROR;
    }
    av_assert0(1 == is_tensor);

    return DNN_SUCCESS;
}

static DNNReturnType get_output_tensor(void *model, DNNData * const output,
                                       OrtValue *output_tensor)
{
    ORTModel *ort_model = (ORTModel *)model;
    ORTContext *ctx = &ort_model->ctx;
    const OrtApi *ort = ort_model->ort;
    OrtStatus **status = &ort_model->status;
    OrtTypeInfo *type = NULL;
    const OrtTensorTypeAndShapeInfo *tensor = NULL;
    ONNXTensorElementDataType data_type;
    int64_t dims[4];
    size_t num_inputs = 0;
    size_t num_dims = 0;

    *status = ort->SessionGetOutputTypeInfo(ort_model->session, 0, &type);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "SessionGetOutputTypeInfo(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_ERROR;
    }
    *status = ort->CastTypeInfoToTensorInfo(type, &tensor);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "CastTypeInfoToTensorInfo(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_ERROR;
    }

    *status = ort->GetTensorElementType(tensor, &data_type);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "GetTensorElementType(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_ERROR;
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
        return DNN_ERROR;
    }

    *status = ort->GetDimensionsCount(tensor, &num_dims);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "GetDimensionsCount(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_ERROR;
    }
    av_assert0(4 == num_dims);

    *status = ort->GetDimensions(tensor, dims, num_dims);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "GetDimensionsCount(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_ERROR;
    }

    ort->ReleaseTypeInfo(type);

    av_assert0(dims[0] == 1);
    output->channels = dims[1];
    output->height = dims[2];
    output->width = dims[3];

    *status = ort->GetTensorMutableData(output_tensor, &output->data);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "GetTensorMutableData(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_ERROR;
    }

    return DNN_SUCCESS;
}

static DNNReturnType get_input_ort(void *model, DNNData *input,
                                   const char *input_name)
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
        return DNN_ERROR;
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
            return DNN_ERROR;
        }
        str_cmp = strncmp(input_name, index_name, 1024);
        *status = ort->AllocatorFree(ort_model->allocator, index_name);
        if (*status) {
            av_log(ctx, AV_LOG_ERROR, "AllocatorFree(): %s\n",
                   ort->GetErrorMessage(*status));
            ort->ReleaseStatus(*status);
            return DNN_ERROR;
        }
        if (0 == str_cmp) {
            index = i;
            break;
        }
    }
    if (-1 == index) {
        av_log(ctx, AV_LOG_ERROR, "Could not find \"%s\" in model\n", input_name);
        return DNN_ERROR;
    }

    *status = ort->SessionGetInputTypeInfo(ort_model->session, index, &type);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "SessionGetInputTypeInfo(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_ERROR;
    }
    *status = ort->CastTypeInfoToTensorInfo(type, &tensor);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "CastTypeInfoToTensorInfo(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_ERROR;
    }

    *status = ort->GetTensorElementType(tensor, &data_type);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "GetTensorElementType(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_ERROR;
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
        return DNN_ERROR;
    }

    *status = ort->GetDimensionsCount(tensor, &num_dims);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "GetDimensionsCount(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_ERROR;
    }
    av_assert0(4 == num_dims);

    *status = ort->GetDimensions(tensor, dims, num_dims);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "GetDimensionsCount(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_ERROR;
    }

    ort->ReleaseTypeInfo(type);

    av_assert0(dims[0] == 1);
    input->channels = dims[1];
    input->height = dims[2];
    input->width = dims[3];

    return DNN_SUCCESS;
}

static DNNReturnType get_output_ort(void *model, const char *input_name,
                                    int input_width, int input_height,
                                    const char *output_name,
                                    int *output_width, int *output_height)
{
    DNNReturnType ret;
    ORTModel *ort_model = (ORTModel *)model;
    ORTContext *ctx = &ort_model->ctx;
    AVFrame *in_frame = av_frame_alloc();
    AVFrame *out_frame = NULL;

    if (!in_frame) {
        av_log(ctx, AV_LOG_ERROR, "Failed to allocate memory for input frame\n");
        return DNN_ERROR;
    }

    out_frame = av_frame_alloc();
    if (!out_frame) {
        av_log(ctx, AV_LOG_ERROR, "Failed to allocate memory for output frame\n");
        av_frame_free(&in_frame);
        return DNN_ERROR;
    }

    in_frame->width = input_width;
    in_frame->height = input_height;

    ret = execute_model_ort(ort_model->model, input_name, in_frame,
                            &output_name, 1, out_frame, 0);
    *output_width = out_frame->width;
    *output_height = out_frame->height;

    av_frame_free(&out_frame);
    av_frame_free(&in_frame);
    return ret;
}

static DNNReturnType load_ort_model(ORTModel *ort_model,
                                    const char *model_filename)
{
    ORTContext *ctx = &ort_model->ctx;
    const OrtApi *ort = NULL;
    OrtStatus **status = &ort_model->status;
    const OrtApiBase *oab = NULL;

    oab = OrtGetApiBase();
    if (NULL == oab) {
        av_log(ctx, AV_LOG_ERROR, "error calling OrtGetApiBase()\n");
        return DNN_ERROR;
    }

    ort_model->ort = oab->GetApi(ORT_API_VERSION);
    if (NULL == ort) {
        av_log(ctx, AV_LOG_ERROR, "error calling GetApi()\n");
        return DNN_ERROR;
    }
    ort = ort_model->ort;

    *status = ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ort_inference",
                             &ort_model->env);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "CreateEnv(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_ERROR;
    }

    *status = ort->CreateSessionOptions(&ort_model->options);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "CreateSessionOptions(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_ERROR;
    }

    *status = ort->SetSessionExecutionMode(ort_model->options,
                                           ctx->options.execute_mode);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "SetSessionExecutionMode(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_ERROR;
    }

    if (ctx->options.profile_file_prefix) {
        *status = ort->EnableProfiling(ort_model->options,
                                       ctx->options.profile_file_prefix);
        if (*status) {
            av_log(ctx, AV_LOG_ERROR, "EnableProfiling(): %s\n",
                   ort->GetErrorMessage(*status));
            ort->ReleaseStatus(*status);
            return DNN_ERROR;
        }
    } else {
        *status = ort->DisableProfiling(ort_model->options);
        if (*status) {
            av_log(ctx, AV_LOG_ERROR, "DisableProfiling(): %s\n",
                   ort->GetErrorMessage(*status));
            ort->ReleaseStatus(*status);
            return DNN_ERROR;
        }
    }

    if (ctx->options.logger_id) {
        *status = ort->SetSessionLogId(ort_model->options,
                                       ctx->options.logger_id);
        if (*status) {
            av_log(ctx, AV_LOG_ERROR, "SetSessionLogId(): %s\n",
                   ort->GetErrorMessage(*status));
            ort->ReleaseStatus(*status);
            return DNN_ERROR;
        }
    }

    *status = ort->SetSessionLogVerbosityLevel(ort_model->options,
                                               ctx->options.log_verbosity_level);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "SetSessionLogVerbosityLevel(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_ERROR;
    }

    *status = ort->SetSessionLogSeverityLevel(ort_model->options,
                                              ctx->options.log_severity_level);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "SetSessionLogSeverityLevel(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_ERROR;
    }

    *status = ort->SetSessionGraphOptimizationLevel(ort_model->options,
                                                    ctx->options.opt_level);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "SetSessionGraphOptimizationLevel(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_ERROR;
    }

    *status = ort->SetIntraOpNumThreads(ort_model->options,
                                        ctx->options.intra_op_threads);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "SetIntraOpNumThreads(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_ERROR;
    }

    *status = ort->SetInterOpNumThreads(ort_model->options,
                                        ctx->options.inter_op_threads);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "SetInterOpNumThreads(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_ERROR;
    }

    *status = ort->GetAllocatorWithDefaultOptions(&ort_model->allocator);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "GetAllocatorWithDefaultOptions(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_ERROR;
    }

    *status = ort->CreateSession(ort_model->env, model_filename,
                                 ort_model->options, &ort_model->session);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "CreateSession(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_ERROR;
    }

    return DNN_SUCCESS;
}

#if 0
#define NAME_BUFFER_SIZE 256

static DNNReturnType add_conv_layer(ORTModel *ort_model, TF_Operation *transpose_op, TF_Operation **cur_op,
                                    ConvolutionalParams* params, const int layer)
{
    ORTContext *ctx = &ort_model->ctx;
    TF_Operation *op;
    TF_OperationDescription *op_desc;
    TF_Output input;
    int64_t strides[] = {1, 1, 1, 1};
    TF_Tensor *tensor;
    int64_t dims[4];
    int dims_len;
    char name_buffer[NAME_BUFFER_SIZE];
    int32_t size;

    size = params->input_num * params->output_num * params->kernel_size * params->kernel_size;
    input.index = 0;

    snprintf(name_buffer, NAME_BUFFER_SIZE, "conv_kernel%d", layer);
    op_desc = TF_NewOperation(ort_model->graph, "Const", name_buffer);
    TF_SetAttrType(op_desc, "dtype", TF_FLOAT);
    dims[0] = params->output_num;
    dims[1] = params->kernel_size;
    dims[2] = params->kernel_size;
    dims[3] = params->input_num;
    dims_len = 4;
    tensor = TF_AllocateTensor(TF_FLOAT, dims, dims_len, size * sizeof(float));
    memcpy(TF_TensorData(tensor), params->kernel, size * sizeof(float));
    TF_SetAttrTensor(op_desc, "value", tensor, ort_model->status);
    if (TF_GetCode(ort_model->status) != TF_OK){
        av_log(ctx, AV_LOG_ERROR, "Failed to set value for kernel of conv layer %d\n", layer);
        return DNN_ERROR;
    }
    op = TF_FinishOperation(op_desc, ort_model->status);
    if (TF_GetCode(ort_model->status) != TF_OK){
        av_log(ctx, AV_LOG_ERROR, "Failed to add kernel to conv layer %d\n", layer);
        return DNN_ERROR;
    }

    snprintf(name_buffer, NAME_BUFFER_SIZE, "transpose%d", layer);
    op_desc = TF_NewOperation(ort_model->graph, "Transpose", name_buffer);
    input.oper = op;
    TF_AddInput(op_desc, input);
    input.oper = transpose_op;
    TF_AddInput(op_desc, input);
    TF_SetAttrType(op_desc, "T", TF_FLOAT);
    TF_SetAttrType(op_desc, "Tperm", TF_INT32);
    op = TF_FinishOperation(op_desc, ort_model->status);
    if (TF_GetCode(ort_model->status) != TF_OK){
        av_log(ctx, AV_LOG_ERROR, "Failed to add transpose to conv layer %d\n", layer);
        return DNN_ERROR;
    }

    snprintf(name_buffer, NAME_BUFFER_SIZE, "conv2d%d", layer);
    op_desc = TF_NewOperation(ort_model->graph, "Conv2D", name_buffer);
    input.oper = *cur_op;
    TF_AddInput(op_desc, input);
    input.oper = op;
    TF_AddInput(op_desc, input);
    TF_SetAttrType(op_desc, "T", TF_FLOAT);
    TF_SetAttrIntList(op_desc, "strides", strides, 4);
    TF_SetAttrString(op_desc, "padding", "VALID", 5);
    *cur_op = TF_FinishOperation(op_desc, ort_model->status);
    if (TF_GetCode(ort_model->status) != TF_OK){
        av_log(ctx, AV_LOG_ERROR, "Failed to add conv2d to conv layer %d\n", layer);
        return DNN_ERROR;
    }

    snprintf(name_buffer, NAME_BUFFER_SIZE, "conv_biases%d", layer);
    op_desc = TF_NewOperation(ort_model->graph, "Const", name_buffer);
    TF_SetAttrType(op_desc, "dtype", TF_FLOAT);
    dims[0] = params->output_num;
    dims_len = 1;
    tensor = TF_AllocateTensor(TF_FLOAT, dims, dims_len, params->output_num * sizeof(float));
    memcpy(TF_TensorData(tensor), params->biases, params->output_num * sizeof(float));
    TF_SetAttrTensor(op_desc, "value", tensor, ort_model->status);
    if (TF_GetCode(ort_model->status) != TF_OK){
        av_log(ctx, AV_LOG_ERROR, "Failed to set value for conv_biases of conv layer %d\n", layer);
        return DNN_ERROR;
    }
    op = TF_FinishOperation(op_desc, ort_model->status);
    if (TF_GetCode(ort_model->status) != TF_OK){
        av_log(ctx, AV_LOG_ERROR, "Failed to add conv_biases to conv layer %d\n", layer);
        return DNN_ERROR;
    }

    snprintf(name_buffer, NAME_BUFFER_SIZE, "bias_add%d", layer);
    op_desc = TF_NewOperation(ort_model->graph, "BiasAdd", name_buffer);
    input.oper = *cur_op;
    TF_AddInput(op_desc, input);
    input.oper = op;
    TF_AddInput(op_desc, input);
    TF_SetAttrType(op_desc, "T", TF_FLOAT);
    *cur_op = TF_FinishOperation(op_desc, ort_model->status);
    if (TF_GetCode(ort_model->status) != TF_OK){
        av_log(ctx, AV_LOG_ERROR, "Failed to add bias_add to conv layer %d\n", layer);
        return DNN_ERROR;
    }

    snprintf(name_buffer, NAME_BUFFER_SIZE, "activation%d", layer);
    switch (params->activation){
    case RELU:
        op_desc = TF_NewOperation(ort_model->graph, "Relu", name_buffer);
        break;
    case TANH:
        op_desc = TF_NewOperation(ort_model->graph, "Tanh", name_buffer);
        break;
    case SIGMOID:
        op_desc = TF_NewOperation(ort_model->graph, "Sigmoid", name_buffer);
        break;
    default:
        av_log(ctx, AV_LOG_ERROR, "Unsupported convolutional activation function\n");
        return DNN_ERROR;
    }
    input.oper = *cur_op;
    TF_AddInput(op_desc, input);
    TF_SetAttrType(op_desc, "T", TF_FLOAT);
    *cur_op = TF_FinishOperation(op_desc, ort_model->status);
    if (TF_GetCode(ort_model->status) != TF_OK){
        av_log(ctx, AV_LOG_ERROR, "Failed to add activation function to conv layer %d\n", layer);
        return DNN_ERROR;
    }

    return DNN_SUCCESS;
}

static DNNReturnType add_depth_to_space_layer(ORTModel *ort_model, TF_Operation **cur_op,
                                              DepthToSpaceParams *params, const int layer)
{
    ORTContext *ctx = &ort_model->ctx;
    TF_OperationDescription *op_desc;
    TF_Output input;
    char name_buffer[NAME_BUFFER_SIZE];

    snprintf(name_buffer, NAME_BUFFER_SIZE, "depth_to_space%d", layer);
    op_desc = TF_NewOperation(ort_model->graph, "DepthToSpace", name_buffer);
    input.oper = *cur_op;
    input.index = 0;
    TF_AddInput(op_desc, input);
    TF_SetAttrType(op_desc, "T", TF_FLOAT);
    TF_SetAttrInt(op_desc, "block_size", params->block_size);
    *cur_op = TF_FinishOperation(op_desc, ort_model->status);
    if (TF_GetCode(ort_model->status) != TF_OK){
        av_log(ctx, AV_LOG_ERROR, "Failed to add depth_to_space to layer %d\n", layer);
        return DNN_ERROR;
    }

    return DNN_SUCCESS;
}

static DNNReturnType add_pad_layer(ORTModel *ort_model, TF_Operation **cur_op,
                                              LayerPadParams *params, const int layer)
{
    ORTContext *ctx = &ort_model->ctx;
    TF_Operation *op;
    TF_Tensor *tensor;
    TF_OperationDescription *op_desc;
    TF_Output input;
    int32_t *pads;
    int64_t pads_shape[] = {4, 2};

    char name_buffer[NAME_BUFFER_SIZE];
    snprintf(name_buffer, NAME_BUFFER_SIZE, "pad%d", layer);

    op_desc = TF_NewOperation(ort_model->graph, "Const", name_buffer);
    TF_SetAttrType(op_desc, "dtype", TF_INT32);
    tensor = TF_AllocateTensor(TF_INT32, pads_shape, 2, 4 * 2 * sizeof(int32_t));
    pads = (int32_t *)TF_TensorData(tensor);
    pads[0] = params->paddings[0][0];
    pads[1] = params->paddings[0][1];
    pads[2] = params->paddings[1][0];
    pads[3] = params->paddings[1][1];
    pads[4] = params->paddings[2][0];
    pads[5] = params->paddings[2][1];
    pads[6] = params->paddings[3][0];
    pads[7] = params->paddings[3][1];
    TF_SetAttrTensor(op_desc, "value", tensor, ort_model->status);
    if (TF_GetCode(ort_model->status) != TF_OK){
        av_log(ctx, AV_LOG_ERROR, "Failed to set value for pad of layer %d\n", layer);
        return DNN_ERROR;
    }
    op = TF_FinishOperation(op_desc, ort_model->status);
    if (TF_GetCode(ort_model->status) != TF_OK){
        av_log(ctx, AV_LOG_ERROR, "Failed to add pad to layer %d\n", layer);
        return DNN_ERROR;
    }

    op_desc = TF_NewOperation(ort_model->graph, "MirrorPad", "mirror_pad");
    input.oper = *cur_op;
    input.index = 0;
    TF_AddInput(op_desc, input);
    input.oper = op;
    TF_AddInput(op_desc, input);
    TF_SetAttrType(op_desc, "T", TF_FLOAT);
    TF_SetAttrType(op_desc, "Tpaddings", TF_INT32);
    TF_SetAttrString(op_desc, "mode", "SYMMETRIC", 9);
    *cur_op = TF_FinishOperation(op_desc, ort_model->status);
    if (TF_GetCode(ort_model->status) != TF_OK){
        av_log(ctx, AV_LOG_ERROR, "Failed to add mirror_pad to layer %d\n", layer);
        return DNN_ERROR;
    }

    return DNN_SUCCESS;
}

static DNNReturnType add_maximum_layer(ORTModel *ort_model, TF_Operation **cur_op,
                                       DnnLayerMaximumParams *params, const int layer)
{
    ORTContext *ctx = &ort_model->ctx;
    TF_Operation *op;
    TF_Tensor *tensor;
    TF_OperationDescription *op_desc;
    TF_Output input;
    float *y;

    char name_buffer[NAME_BUFFER_SIZE];
    snprintf(name_buffer, NAME_BUFFER_SIZE, "maximum/y%d", layer);

    op_desc = TF_NewOperation(ort_model->graph, "Const", name_buffer);
    TF_SetAttrType(op_desc, "dtype", TF_FLOAT);
    tensor = TF_AllocateTensor(TF_FLOAT, NULL, 0, TF_DataTypeSize(TF_FLOAT));
    y = (float *)TF_TensorData(tensor);
    *y = params->val.y;
    TF_SetAttrTensor(op_desc, "value", tensor, ort_model->status);
    if (TF_GetCode(ort_model->status) != TF_OK){
        av_log(ctx, AV_LOG_ERROR, "Failed to set value for maximum/y of layer %d", layer);
        return DNN_ERROR;
    }
    op = TF_FinishOperation(op_desc, ort_model->status);
    if (TF_GetCode(ort_model->status) != TF_OK){
        av_log(ctx, AV_LOG_ERROR, "Failed to add maximum/y to layer %d\n", layer);
        return DNN_ERROR;
    }

    snprintf(name_buffer, NAME_BUFFER_SIZE, "maximum%d", layer);
    op_desc = TF_NewOperation(ort_model->graph, "Maximum", name_buffer);
    input.oper = *cur_op;
    input.index = 0;
    TF_AddInput(op_desc, input);
    input.oper = op;
    TF_AddInput(op_desc, input);
    TF_SetAttrType(op_desc, "T", TF_FLOAT);
    *cur_op = TF_FinishOperation(op_desc, ort_model->status);
    if (TF_GetCode(ort_model->status) != TF_OK){
        av_log(ctx, AV_LOG_ERROR, "Failed to add maximum to layer %d\n", layer);
        return DNN_ERROR;
    }

    return DNN_SUCCESS;
}

static DNNReturnType load_native_model(ORTModel *ort_model, const char *model_filename)
{
    ORTContext *ctx = &ort_model->ctx;
    int32_t layer;
    TF_OperationDescription *op_desc;
    TF_Operation *op;
    TF_Operation *transpose_op;
    TF_Tensor *tensor;
    TF_Output input;
    int32_t *transpose_perm;
    int64_t transpose_perm_shape[] = {4};
    int64_t input_shape[] = {1, -1, -1, -1};
    DNNReturnType layer_add_res;
    DNNModel *model = NULL;
    NativeModel *native_model;

    model = ff_dnn_load_model_native(model_filename, NULL, NULL);
    if (!model){
        av_log(ctx, AV_LOG_ERROR, "Failed to load native model\n");
        return DNN_ERROR;
    }

    native_model = (NativeModel *)model->model;
    ort_model->graph = TF_NewGraph();
    ort_model->status = TF_NewStatus();

#define CLEANUP_ON_ERROR(ort_model) \
    { \
        TF_DeleteGraph(ort_model->graph); \
        TF_DeleteStatus(ort_model->status); \
        av_log(ctx, AV_LOG_ERROR, "Failed to set value or add operator to layer\n"); \
        return DNN_ERROR; \
    }

    op_desc = TF_NewOperation(ort_model->graph, "Placeholder", "x");
    TF_SetAttrType(op_desc, "dtype", TF_FLOAT);
    TF_SetAttrShape(op_desc, "shape", input_shape, 4);
    op = TF_FinishOperation(op_desc, ort_model->status);
    if (TF_GetCode(ort_model->status) != TF_OK){
        CLEANUP_ON_ERROR(ort_model);
    }

    op_desc = TF_NewOperation(ort_model->graph, "Const", "transpose_perm");
    TF_SetAttrType(op_desc, "dtype", TF_INT32);
    tensor = TF_AllocateTensor(TF_INT32, transpose_perm_shape, 1, 4 * sizeof(int32_t));
    transpose_perm = (int32_t *)TF_TensorData(tensor);
    transpose_perm[0] = 1;
    transpose_perm[1] = 2;
    transpose_perm[2] = 3;
    transpose_perm[3] = 0;
    TF_SetAttrTensor(op_desc, "value", tensor, ort_model->status);
    if (TF_GetCode(ort_model->status) != TF_OK){
        CLEANUP_ON_ERROR(ort_model);
    }
    transpose_op = TF_FinishOperation(op_desc, ort_model->status);

    for (layer = 0; layer < native_model->layers_num; ++layer){
        switch (native_model->layers[layer].type){
        case DLT_INPUT:
            layer_add_res = DNN_SUCCESS;
            break;
        case DLT_CONV2D:
            layer_add_res = add_conv_layer(ort_model, transpose_op, &op,
                                           (ConvolutionalParams *)native_model->layers[layer].params, layer);
            break;
        case DLT_DEPTH_TO_SPACE:
            layer_add_res = add_depth_to_space_layer(ort_model, &op,
                                                     (DepthToSpaceParams *)native_model->layers[layer].params, layer);
            break;
        case DLT_MIRROR_PAD:
            layer_add_res = add_pad_layer(ort_model, &op,
                                          (LayerPadParams *)native_model->layers[layer].params, layer);
            break;
        case DLT_MAXIMUM:
            layer_add_res = add_maximum_layer(ort_model, &op,
                                          (DnnLayerMaximumParams *)native_model->layers[layer].params, layer);
            break;
        default:
            CLEANUP_ON_ERROR(ort_model);
        }

        if (layer_add_res != DNN_SUCCESS){
            CLEANUP_ON_ERROR(ort_model);
        }
    }

    op_desc = TF_NewOperation(ort_model->graph, "Identity", "y");
    input.oper = op;
    input.index = 0;
    TF_AddInput(op_desc, input);
    TF_FinishOperation(op_desc, ort_model->status);
    if (TF_GetCode(ort_model->status) != TF_OK){
        CLEANUP_ON_ERROR(ort_model);
    }

    ff_dnn_free_model_native(&model);

    return DNN_SUCCESS;
}
#endif

DNNModel *ff_dnn_load_model_ort(const char *model_filename,
                                const char *options, void *userdata)
{
    DNNModel *model = NULL;
    ORTModel *ort_model = NULL;

    model = av_mallocz(sizeof(DNNModel));
    if (!model) {
        return NULL;
    }

    ort_model = av_mallocz(sizeof(ORTModel));
    if (!ort_model) {
        av_freep(&model);
        return NULL;
    }
    ort_model->ctx.class = &dnn_onnxruntime_class;
    ort_model->model = model;

    //parse options
    av_opt_set_defaults(&ort_model->ctx);
    if (av_opt_set_from_string(&ort_model->ctx, options, NULL, "=", "&") < 0) {
        av_log(&ort_model->ctx, AV_LOG_ERROR,
               "Failed to parse options \"%s\"\n", options);
        av_freep(&ort_model);
        av_freep(&model);
        return NULL;
    }

    if (load_ort_model(ort_model, model_filename) != DNN_SUCCESS) {
#if 1
        av_freep(&ort_model);
        av_freep(&model);

        return NULL;
#else
        if (load_native_model(ort_model, model_filename) != DNN_SUCCESS) {
            av_freep(&ort_model);
            av_freep(&model);

            return NULL;
        }
#endif
    }

    model->model = (void *)ort_model;
    model->get_input = &get_input_ort;
    model->get_output = &get_output_ort;
    model->options = options;
    model->userdata = userdata;

    return model;
}

static DNNReturnType execute_model_ort(const DNNModel *model,
                                       const char *input_name,
                                       AVFrame *in_frame,
                                       const char **output_names,
                                       uint32_t nb_output,
                                       AVFrame *out_frame,
                                       int do_ioproc)
{
    ORTModel *ort_model = (ORTModel *)model->model;
    ORTContext *ctx = &ort_model->ctx;
    const OrtApi *ort = ort_model->ort;
    size_t num_outputs = 0;
    OrtStatus **status = &ort_model->status;
    OrtValue *input_tensor = NULL;
    OrtValue *output_tensor = NULL;
    int is_tensor = 0;
    DNNData input, output;

    if (get_input_ort(ort_model, &input, input_name) != DNN_SUCCESS) {
        return DNN_ERROR;
    }
    input.height = in_frame->height;
    input.width = in_frame->width;

    if (DNN_SUCCESS != allocate_input_tensor(model, &input, &input_tensor)) {
        av_log(ctx, AV_LOG_ERROR, "Cannot allocate memory for input tensor\n");
        return DNN_ERROR;
    }

    *status = ort->GetTensorMutableData(input_tensor, &input.data);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "GetTensorMutableData(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_ERROR;
    }

    if (do_ioproc) {
        if (ort_model->model->pre_proc != NULL) {
            ort_model->model->pre_proc(in_frame, &input,
                                       ort_model->model->userdata);
        } else {
            proc_from_frame_to_dnn(in_frame, &input, ctx);
        }
    }

    if (nb_output != 1) {
        // currently, the filter does not need multiple outputs,
        // so we just pending the support until we really need it.
        av_log(ctx, AV_LOG_ERROR, "Do not support multiple outputs\n");
        return DNN_ERROR;
    }

    *status = ort->SessionGetOutputCount(ort_model->session, &num_outputs);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "SessionGetOutputCount(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_ERROR;
    }

    if (num_outputs != 1) {
        // currently, follow the given outputs
        av_log(ctx, AV_LOG_ERROR, "Model has multiple outputs\n");
        return DNN_ERROR;
    }

    *status = ort->Run(ort_model->session, NULL, &input_name,
                       (const OrtValue * const *)&input_tensor, 1,
                       output_names, 1, &output_tensor);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "Run(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_ERROR;
    }
    *status = ort->IsTensor(output_tensor, &is_tensor);
    if (*status) {
        av_log(ctx, AV_LOG_ERROR, "IsTensor(): %s\n",
               ort->GetErrorMessage(*status));
        ort->ReleaseStatus(*status);
        return DNN_ERROR;
    }
    av_assert0(1 == is_tensor);

    if (DNN_SUCCESS != get_output_tensor(ort_model, &output, output_tensor)) {
        av_log(ctx, AV_LOG_ERROR, "Cannot get output\n");
        return DNN_ERROR;
    }

    if (do_ioproc) {
        if (ort_model->model->post_proc != NULL) {
            ort_model->model->post_proc(out_frame, &output,
                                        ort_model->model->userdata);
        } else {
            proc_from_dnn_to_frame(out_frame, &output, ctx);
        }
    } else {
        out_frame->width = output.width;
        out_frame->height = output.height;
    }

    ort->ReleaseValue(input_tensor);
    ort->ReleaseValue(output_tensor);

    return DNN_SUCCESS;
}

DNNReturnType ff_dnn_execute_model_ort(const DNNModel *model,
                                       const char *input_name,
                                       AVFrame *in_frame,
                                       const char **output_names,
                                       uint32_t nb_output,
                                       AVFrame *out_frame)
{
    ORTModel *ort_model = (ORTModel *)model->model;
    ORTContext *ctx = &ort_model->ctx;

    if (!in_frame) {
        av_log(ctx, AV_LOG_ERROR, "In frame is NULL when execute model.\n");
        return DNN_ERROR;
    }

    if (!out_frame) {
        av_log(ctx, AV_LOG_ERROR, "Out frame is NULL when execute model.\n");
        return DNN_ERROR;
    }

    return execute_model_ort(model, input_name, in_frame, output_names,
                             nb_output, out_frame, 1);
}

void ff_dnn_free_model_ort(DNNModel **model)
{
    ORTModel *ort_model;

    if (*model) {
        ort_model = (ORTModel *)(*model)->model;
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

        av_freep(&ort_model);
        av_freep(model);
    }
}


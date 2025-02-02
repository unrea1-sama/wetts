// Copyright (c) 2022 Zhendong Peng (pzd17@tsinghua.org.cn)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "model/onnx_tts_model.h"

#include <utility>

#include "glog/logging.h"

namespace wetts {

Ort::Env OnnxTtsModel::env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "");
Ort::SessionOptions OnnxTtsModel::session_options_ = Ort::SessionOptions();

void OnnxTtsModel::InitEngineThreads(int num_threads) {
  session_options_.SetIntraOpNumThreads(num_threads);
}

void OnnxTtsModel::GetInputOutputInfo(
    const std::shared_ptr<Ort::Session>& session,
    std::vector<const char*>* in_names, std::vector<const char*>* out_names) {
  Ort::AllocatorWithDefaultOptions allocator;
  // Input info
  int num_nodes = session->GetInputCount();
  in_names->resize(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    char* name = session->GetInputName(i, allocator);
    Ort::TypeInfo type_info = session->GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = tensor_info.GetElementType();
    std::vector<int64_t> node_dims = tensor_info.GetShape();
    std::stringstream shape;
    for (auto j : node_dims) {
      shape << j;
      shape << " ";
    }
    LOG(INFO) << "\tInput " << i << " : name=" << name << " type=" << type
              << " dims=" << shape.str();
    (*in_names)[i] = name;
  }
  // Output info
  num_nodes = session->GetOutputCount();
  out_names->resize(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    char* name = session->GetOutputName(i, allocator);
    Ort::TypeInfo type_info = session->GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = tensor_info.GetElementType();
    std::vector<int64_t> node_dims = tensor_info.GetShape();
    std::stringstream shape;
    for (auto j : node_dims) {
      shape << j;
      shape << " ";
    }
    LOG(INFO) << "\tOutput " << i << " : name=" << name << " type=" << type
              << " dims=" << shape.str();
    (*out_names)[i] = name;
  }
}

void OnnxTtsModel::Read(const std::string& model_path) {
  // 1. Load sessions
  try {
#ifdef _MSC_VER
    session_ = std::make_shared<Ort::Session>(
        env_, ToWString(model_path).c_str(), session_options_);
#else
    session_ = std::make_shared<Ort::Session>(env_, model_path.c_str(),
                                              session_options_);
#endif
  } catch (std::exception const& e) {
    LOG(ERROR) << "error when load onnx model: " << e.what();
    exit(0);
  }

  // 2. TODO: Read metadata
  // auto model_metadata = session_->GetModelMetadata();
  // Ort::AllocatorWithDefaultOptions allocator;
  // sampling_rate_ =
  //     atoi(model_metadata.LookupCustomMetadataMap("sampling_rate",
  //     allocator));
  // LOG(INFO) << "Onnx Model Info:";
  // LOG(INFO) << "\tsampling_rate " << sampling_rate_;

  // 3. Read model nodes
  GetInputOutputInfo(session_, &in_names_, &out_names_);
}

void OnnxTtsModel::Forward(std::vector<int64_t>* phonemes,
                           std::vector<float>* audio) {
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

  int num_phones = phonemes->size();
  const int64_t inputs_shape[] = {1, num_phones};
  auto inputs_ort = Ort::Value::CreateTensor<int64_t>(
      memory_info, phonemes->data(), num_phones, inputs_shape, 2);

  std::vector<int64_t> inputs_len = {num_phones};
  const int64_t inputs_len_shape[] = {1};
  auto inputs_len_ort = Ort::Value::CreateTensor<int64_t>(
      memory_info, inputs_len.data(), inputs_len.size(), inputs_len_shape, 1);

  std::vector<float> scales = {0.667, 1.0, 0.8};
  const int64_t scales_shape[] = {1, 3};
  auto scales_ort = Ort::Value::CreateTensor<float>(
      memory_info, scales.data(), scales.size(), scales_shape, 2);

  std::vector<Ort::Value> ort_inputs;
  ort_inputs.push_back(std::move(inputs_ort));
  ort_inputs.push_back(std::move(inputs_len_ort));
  ort_inputs.push_back(std::move(scales_ort));

  auto outputs_ort =
      session_->Run(Ort::RunOptions{nullptr}, in_names_.data(),
                    ort_inputs.data(), ort_inputs.size(), out_names_.data(), 1);
  int len = outputs_ort[0].GetTensorTypeAndShapeInfo().GetShape()[2];
  float* outputs = outputs_ort[0].GetTensorMutableData<float>();
  audio->assign(outputs, outputs + len);
}

}  // namespace wetts

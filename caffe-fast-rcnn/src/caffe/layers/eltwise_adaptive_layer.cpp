#include <cfloat>
#include <vector>

#include "caffe/layers/eltwise_adaptive_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EltwiseAdaptiveLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(this->layer_param().eltwise_param().coeff_size() == 0
      || this->layer_param().eltwise_param().coeff_size() == bottom.size()) <<
      "Eltwise Layer takes one coefficient per bottom blob.";
  CHECK(!(this->layer_param().eltwise_param().operation()
      == EltwiseParameter_EltwiseOp_PROD
      && this->layer_param().eltwise_param().coeff_size())) <<
      "Eltwise layer only takes coefficients for summation.";
  op_ = this->layer_param_.eltwise_param().operation();
  // Blob-wise coefficients for the elementwise operation.
  coeffs_ = vector<Dtype>(bottom.size(), 1);
  if (this->layer_param().eltwise_param().coeff_size()) {
    for (int i = 0; i < bottom.size(); ++i) {
      coeffs_[i] = this->layer_param().eltwise_param().coeff(i);
    }
  }
  stable_prod_grad_ = this->layer_param_.eltwise_param().stable_prod_grad();

  LayerParameter crop_layer_param;
  //caffe::CropParameter* crop_param = new caffe::CropParameter();
  crop_layer_param.set_name("EleWise_Crop_");
  crop_layer_param.set_type("Crop");
  crop_param = crop_layer_param.mutable_crop_param();
  int crop_axis = -1;
  bool crop_axis_begin = false;
  //std::cout << "---------------------------------------";
  //const LayerParameter& crop_layer_param = crop_layer_->layer_param();
  //caffe::CropParameter* crop_param = ((LayerParameter&)crop_layer_param).mutable_crop_param();
  for(int i = 0; i < bottom[0]->shape().size();i++)
  {
    if(bottom[0]->shape(i)!=bottom[1]->shape(i))
    {
      //std::cout << "offset index ----------------" << i;
      int offset= bottom[0]->shape(i)-bottom[1]->shape(i);
      if(crop_axis_begin == false){
        crop_axis_begin = true;
        crop_param->set_axis(i);
        crop_param->add_offset(offset);
      }else{
        crop_param->add_offset(offset);
      }
    }
    else{
      if(crop_axis_begin == true){
        crop_param->add_offset(0);
      }
    }
  }
    
  crop_bottom_vec_.clear();
  crop_bottom_vec_.push_back(bottom[0]);
  crop_bottom_vec_.push_back(bottom[1]);
  crop_top_vec_.clear();
  crop_top_vec_.push_back(&crop_top_blob_);
  crop_layer_ = LayerRegistry<Dtype>::CreateLayer(crop_layer_param);
  crop_layer_->SetUp(crop_bottom_vec_, crop_top_vec_);
}

template <typename Dtype>
void EltwiseAdaptiveLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    //only for sum operation
    //crop_layer_->Reshape(crop_bottom_vec_, crop_top_vec_);
  // for (int i = 1; i < bottom.size(); ++i) {
  //   CHECK(bottom[0]->shape() == bottom[i]->shape())
  //       << "bottom[0]: " << bottom[0]->shape_string()
  //       << ", bottom[" << i << "]: " << bottom[i]->shape_string();
  // }
    // int crop_axis = -1;
    // bool crop_axis_begin = false;
    // std::cout << "test begin-------------------------------***************";
    // int offset_index = -1;
    // //caffe::CropParameter* crop_param = crop_param->mutable_crop_param();
    // for(int i = 0; i < crop_bottom_vec_[0]->shape().size();i++)
    // {
    //   std::cout << "crop size:----------------------------" << crop_bottom_vec_.size();
    //   if(crop_bottom_vec_[0]->shape(i)!=crop_bottom_vec_[1]->shape(i))
    //   {
    //     //std::cout << "offset index ----------------" << i;
    //     int offset= crop_bottom_vec_[0]->shape(i)-crop_bottom_vec_[1]->shape(i);
    //     std::cout << "offset-------" << offset;
    //     if(offset < 0){
    //         Blob<Dtype>* bottom0= crop_bottom_vec_[1];
    //         Blob<Dtype>* bottom1= crop_bottom_vec_[0];
    //         crop_bottom_vec_.clear();
    //         crop_bottom_vec_.push_back(bottom0);
    //         crop_bottom_vec_.push_back(bottom1);
    //         offset = -1*offset;
    //     }
    //     std::cout << "offset2-------" << offset;
    //     if(crop_axis_begin == false){
    //       crop_axis_begin = true;
    //       std::cout << "before clear_offset-------";
    //       crop_param->clear_offset();
    //       std::cout << "after clear_offset-------";
    //       std::cout << "before set_axis-------";
    //       crop_param->set_axis(i);
    //       offset_index =0;
    //       std::cout << "after set_axis-------";
    //       std::cout << "before add_offset-------";
    //       crop_param->mutable_offset()->Set(offset_index,offset);
    //       std::cout << "after add_offset-------";
    //     }else{
    //       offset_index++;
    //       crop_param->mutable_offset()->Set(offset_index,offset);
    //     }
    //   }
    //   else{
    //     std::cout << "offset3-------" << 0;
    //     if(crop_axis_begin == true){
    //       offset_index++;
    //       crop_param->mutable_offset()->Set(offset_index,0);
    //     }
    //   }
    // }
    
    top[0]->ReshapeLike(*crop_bottom_vec_[0]);
    // If max operation, we will initialize the vector index part.
    if (this->layer_param_.eltwise_param().operation() ==
        EltwiseParameter_EltwiseOp_MAX && top.size() == 1) {
      max_idx_.Reshape(bottom[0]->shape());
    }
}

template <typename Dtype>
void EltwiseAdaptiveLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int* mask = NULL;
  const Dtype* bottom_data_a = NULL;
  const Dtype* bottom_data_b = NULL;

  const int count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  switch (op_) {
  case EltwiseParameter_EltwiseOp_PROD:
    caffe_mul(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(), top_data);
    for (int i = 2; i < bottom.size(); ++i) {
      caffe_mul(count, top_data, bottom[i]->cpu_data(), top_data);
    }
    break;
  case EltwiseParameter_EltwiseOp_SUM:
    caffe_set(count, Dtype(0), top_data);
    crop_layer_->Forward(crop_bottom_vec_, crop_top_vec_);
    caffe_gpu_axpy(count, coeffs_[0], crop_top_blob_.cpu_data(), top_data);
    caffe_gpu_axpy(count, coeffs_[1], crop_bottom_vec_[1]->cpu_data(), top_data);
    // caffe_axpy(count, coeffs_[0], crop_top_vec_[0]->cpu_data(), top_data);
    // bottom[0] = boost::shared_ptr<Blob<Dtype>*>.reset((crop_top_blob_);
    // // TODO(shelhamer) does BLAS optimize to sum for coeff = 1?
    // for (int i = 0; i < bottom.size(); ++i) {
    //   caffe_axpy(count, coeffs_[i], bottom[i]->cpu_data(), top_data);
    // }
    break;
  case EltwiseParameter_EltwiseOp_MAX:
    // Initialize
    mask = max_idx_.mutable_cpu_data();
    caffe_set(count, -1, mask);
    caffe_set(count, Dtype(-FLT_MAX), top_data);
    // bottom 0 & 1
    bottom_data_a = bottom[0]->cpu_data();
    bottom_data_b = bottom[1]->cpu_data();
    for (int idx = 0; idx < count; ++idx) {
      if (bottom_data_a[idx] > bottom_data_b[idx]) {
        top_data[idx] = bottom_data_a[idx];  // maxval
        mask[idx] = 0;  // maxid
      } else {
        top_data[idx] = bottom_data_b[idx];  // maxval
        mask[idx] = 1;  // maxid
      }
    }
    // bottom 2++
    for (int blob_idx = 2; blob_idx < bottom.size(); ++blob_idx) {
      bottom_data_b = bottom[blob_idx]->cpu_data();
      for (int idx = 0; idx < count; ++idx) {
        if (bottom_data_b[idx] > top_data[idx]) {
          top_data[idx] = bottom_data_b[idx];  // maxval
          mask[idx] = blob_idx;  // maxid
        }
      }
    }
    break;
  default:
    LOG(FATAL) << "Unknown elementwise operation.";
  }
}

template <typename Dtype>
void EltwiseAdaptiveLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int* mask = NULL;
  const int count = top[0]->count();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->cpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
      switch (op_) {
      case EltwiseParameter_EltwiseOp_PROD:
        if (stable_prod_grad_) {
          bool initialized = false;
          for (int j = 0; j < bottom.size(); ++j) {
            if (i == j) { continue; }
            if (!initialized) {
              caffe_copy(count, bottom[j]->cpu_data(), bottom_diff);
              initialized = true;
            } else {
              caffe_mul(count, bottom[j]->cpu_data(), bottom_diff,
                        bottom_diff);
            }
          }
        } else {
          caffe_div(count, top_data, bottom_data, bottom_diff);
        }
        caffe_mul(count, bottom_diff, top_diff, bottom_diff);
        break;
      case EltwiseParameter_EltwiseOp_SUM:
        if (coeffs_[i] == Dtype(1)) {
          crop_layer_->Backward(crop_top_vec_,propagate_down,crop_bottom_vec_);
          const Dtype* top_diff = crop_bottom_vec_[0]->cpu_diff();
          caffe_copy(count, top_diff, bottom_diff);
        } else {
          caffe_cpu_scale(count, coeffs_[i], top_diff, bottom_diff);
        }
        break;
      case EltwiseParameter_EltwiseOp_MAX:
        mask = max_idx_.cpu_data();
        for (int index = 0; index < count; ++index) {
          Dtype gradient = 0;
          if (mask[index] == i) {
            gradient += top_diff[index];
          }
          bottom_diff[index] = gradient;
        }
        break;
      default:
        LOG(FATAL) << "Unknown elementwise operation.";
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EltwiseAdaptiveLayer);
#endif

INSTANTIATE_CLASS(EltwiseAdaptiveLayer);
REGISTER_LAYER_CLASS(EltwiseAdaptive);

}  // namespace caffe

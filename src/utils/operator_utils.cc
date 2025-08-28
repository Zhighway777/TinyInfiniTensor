#include "utils/operator_utils.h"
#include "core/runtime.h"

namespace infini {

Shape infer_broadcast(const Shape &A, const Shape &B) {

    // =================================== 作业 ===================================
    // TODO：对 A 和 B 进行双向广播，返回广播后的形状。
    // REF: https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
    // =================================== 作业 ===================================
    
    // 处理空张量的情况
    if (A.empty()) return B;
    if (B.empty()) return A;
    
    int rank_A = A.size();
    int rank_B = B.size();
    
    // 确定最终的维度数
    int max_rank = std::max(rank_A, rank_B);
    
    // 创建对齐后的形状
    Shape local_A = A;
    Shape local_B = B;
    
    // 在前面填充1，使两个形状的维度数相同
    if (rank_A < max_rank) {
        local_A.insert(local_A.begin(), max_rank - rank_A, 1);
    }
    if (rank_B < max_rank) {
        local_B.insert(local_B.begin(), max_rank - rank_B, 1);
    }
    
    // 创建结果形状
    Shape result(max_rank);
    
    // 按照广播规则计算每个维度
    for (int i = 0; i < max_rank; i++) {
        int dim_A = local_A[i];
        int dim_B = local_B[i];
        
        if (dim_A == dim_B) {
            // 维度相同，直接使用
            result[i] = dim_A;
        } else if (dim_A == 1) {
            // A 的维度为1，可以广播到 B 的维度
            result[i] = dim_B;
        } else if (dim_B == 1) {
            // B 的维度为1，可以广播到 A 的维度
            result[i] = dim_A;
        } else {
            // 维度不兼容，无法广播
            IT_ASSERT(false, 
                "Cannot broadcast shapes: dimension " + std::to_string(i) + 
                " has incompatible sizes " + std::to_string(dim_A) + 
                " and " + std::to_string(dim_B));
        }
    }
    
    return result;
}

int get_real_axis(const int &axis, const int &rank) {
    IT_ASSERT(rank >= 1);
    IT_ASSERT(axis >= -rank && axis <= (rank - 1));
    int newAxis;
    if (axis < 0) {
        newAxis = rank + axis;
    } else {
        newAxis = axis;
    }
    return newAxis;
}

Shape locate_index(size_t inputN, const Shape &shape) {
    Shape ans(shape.size());
    auto i = ans.rbegin();
    auto j = shape.rbegin(), ej = shape.rend();
    while (j != ej) {
        auto div = std::div(inputN, *j++);
        *i++ = div.rem;
        inputN = div.quot;
    }
    return ans;
}

size_t delocate_index(const Shape &shapeIndex, const Shape &shape,
                      const Shape &stride) {
    size_t ans = 0;
    Shape index(shapeIndex.size());
    IT_ASSERT(shapeIndex.size() == shape.size());
    IT_ASSERT(shape.size() == stride.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        index[i] = shapeIndex[i] % shape[i];
        ans += index[i] * stride[i];
    }
    return ans;
}

std::string device_to_str(Device device) {
    std::string deviceStr;
    switch (device) {
    case Device::CPU:
        return "CPU";
    default:
        IT_TODO_HALT();
    }
}

std::string get_kernel_attrs_str(const KernelAttrs &kernelAttrs) {
    std::string deviceStr = device_to_str(std::get<0>(kernelAttrs));
    std::string opStr = OpType(std::get<1>(kernelAttrs)).toString();
    return deviceStr + ", " + opStr;
}

} // namespace infini

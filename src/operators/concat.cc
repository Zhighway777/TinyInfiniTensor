#include "operators/concat.h"
#include "utils/operator_utils.h"

namespace infini {
ConcatObj::ConcatObj(GraphObj *graph, TensorVec inputs, Tensor output, int _dim)
    : OperatorObj(OpType::Concat, inputs, {output}) {
    int rank = inputs[0]->getRank();
    dim = get_real_axis(_dim, rank);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ConcatObj::inferShape(const TensorVec &inputs) {
    if (inputs.empty()) {
        return std::nullopt;
    }

    Shape dims = inputs[0]->getDims();
    auto rank = inputs[0]->getRank();

    // =================================== 作业 ===================================
    // TODO：修改 dims，返回正确的 concat 后的 shape
    // REF: https://onnx.ai/onnx/operators/onnx__Concat.html#concat-13
    // =================================== 作业 ===================================
    // 预计算拼接维度的大小
    IT_ASSERT(dim > 0 && dim < static_cast<int>(rank), "Dimension out of range");

    int concat_size = dims[dim];
    for (size_t i = 1; i < inputs.size(); i++) {
        IT_ASSERT(inputs[i]->getRank() == rank, "All input tensors must have the same rank");
        for (int j = 0; j < static_cast<int>(rank); j++) {
            if (j != dim){
                IT_ASSERT(dims[j] == inputs[i]->getDims()[j], "All input tensors must have the same shape except the concatenation dimension");
            }
        }

        concat_size += inputs[i]->getDims()[dim];
    }
    dims[dim] = concat_size;
    
    return {{dims}};
}

std::string ConcatObj::toString() const {
    std::ostringstream os;
    os << "Concat[" << getGuid() << "]";
    os << "(";
    for (auto input : inputs)
        os << vecToString(input->getDims()) << ",";
    os << "dim=" << dim << ",";
    os << "input=";
    for (auto input : inputs)
        os << input->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

} // namespace infini

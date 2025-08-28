#include "operators/matmul.h"
#include "utils/operator_utils.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB), m(0), n(0), k(0)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        
        if (inputs.size() != 2) {
            return std::nullopt;
        }
        
        auto A = inputs[0];
        auto B = inputs[1];
        
        if (!A || !B) {
            return std::nullopt;
        }
        
        Shape shapeA = A->getDims();
        Shape shapeB = B->getDims();
        
        if (shapeA.size() < 2 || shapeB.size() < 2) {
            return std::nullopt;
        }
        
        // 获取 A 和 B 的实际矩阵维度（考虑转置）
        Shape effectiveA = shapeA;
        Shape effectiveB = shapeB;
        
        // 处理转置：只影响最后两个维度
        if (transA && effectiveA.size() >= 2) {
            size_t rank = effectiveA.size();
            std::swap(effectiveA[rank-2], effectiveA[rank-1]);
        }
        
        if (transB && effectiveB.size() >= 2) {
            size_t rank = effectiveB.size();
            std::swap(effectiveB[rank-2], effectiveB[rank-1]);
        }
        
        // 获取矩阵乘法的维度
        size_t rankA = effectiveA.size();
        size_t rankB = effectiveB.size();
        
        // 提取最后两个维度进行矩阵乘法
        int m = effectiveA[rankA-2];  // A 的行数
        int k_A = effectiveA[rankA-1];  // A 的列数
        int k_B = effectiveB[rankB-2];  // B 的行数
        int n = effectiveB[rankB-1];  // B 的列数
        
        // 检查矩阵乘法的维度兼容性
        if (k_A != k_B) {
            return std::nullopt;
        }
        
        // 处理批次维度的广播
        Shape batchA(shapeA.begin(), shapeA.end()-2);  // A 的批次维度
        Shape batchB(shapeB.begin(), shapeB.end()-2);  // B 的批次维度
        
        // 广播批次维度
        Shape resultBatch;
        try {
            if (batchA.empty() && batchB.empty()) {
                // 都没有批次维度
                resultBatch = {};
            } else if (batchA.empty()) {
                // A 没有批次维度，使用 B 的批次维度
                resultBatch = batchB;
            } else if (batchB.empty()) {
                // B 没有批次维度，使用 A 的批次维度
                resultBatch = batchA;
            } else {
                // 都有批次维度，需要广播
                resultBatch = infer_broadcast(batchA, batchB);
            }
        } catch (...) {
            return std::nullopt;
        }
        
        // 构造输出形状：批次维度 + [m, n]
        Shape outputShape = resultBatch;
        outputShape.push_back(m);
        outputShape.push_back(n);
        
        // 设置成员变量用于后续计算
        this->m = m;
        this->n = n;
        this->k = k_A;
        
        return {{outputShape}};
    }

} // namespace infini
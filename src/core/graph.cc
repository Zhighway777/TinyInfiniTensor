#include "core/graph.h"
#include "operators/transpose.h"
#include "operators/matmul.h"
#include <algorithm>
#include <numeric>
#include <queue>

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================
    // 使用更安全的方式：先收集要删除的操作符，然后统一删除
    std::vector<std::pair<Operator, Operator>> to_remove;
    
    for (const auto& op : ops) {
        if (op->getOpType() == OpType::Transpose) {
            auto successors = op->getSuccessors();
            
            for (const auto& succ : successors) {
                if (succ->getOpType() == OpType::Transpose) {
                    // 检查是否是逆操作
                    auto transpose1 = dynamic_cast<TransposeObj*>(op.get());
                    auto transpose2 = dynamic_cast<TransposeObj*>(succ.get());
                    
                    if (this->areSameTransposes(transpose1, transpose2)) {
                        to_remove.emplace_back(op, succ);
                        break;  // 找到一个匹配就跳出内层循环
                    }
                }
            }
        }
    }
    
    // 统一处理删除和重新连接
    for (const auto& pair : to_remove) {
        auto op1 = pair.first;
        auto op2 = pair.second;
        
        // 重新连接图结构
        this->reconnectGraph(op1, op2);
        
        // 删除操作符
        this->removeOperator(op1);
        this->removeOperator(op2);
    }
    
    // =================================== 合并算子优化 ===================================
    // 将转置操作融入到矩阵乘算子的属性中
    std::vector<std::pair<Operator, Operator>> to_merge;
    
    for (const auto& op : ops) {
        if (op && op->getOpType() == OpType::Transpose) {
            auto successors = op->getSuccessors();
            
            for (const auto& succ : successors) {
                // 安全检查：确保后继操作符仍然有效
                if (!succ) continue;
                
                if (succ->getOpType() == OpType::MatMul) {
                    // 检查转置是否对最后两个维度做交换
                    auto transpose = dynamic_cast<TransposeObj*>(op.get());
                    auto matmul = dynamic_cast<MatmulObj*>(succ.get());
                    
                    if (transpose && matmul && this->isLastTwoDimsSwap(transpose)) {
                        to_merge.emplace_back(op, succ);
                        break;
                    }
                }
            }
        }
    }
    
    // 统一处理合并
    for (const auto& pair : to_merge) {
        auto transpose = pair.first;
        auto matmul = pair.second;
        
        // 安全检查：确保操作符仍然有效
        if (!transpose || !matmul) continue;
        
        // 合并转置到矩阵乘
        this->mergeTransposeToMatmul(transpose, matmul);
        
        // 删除转置操作符
        this->removeOperator(transpose);
    }
    
    // 清理未使用的张量
    this->cleanupUnusedTensors();

}

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================
        
        // 第一阶段：收集所有tensor的内存需求并分配偏移
        std::unordered_set<TensorObj*> allocated_tensors;
        std::vector<std::pair<Tensor, size_t>> tensor_offsets;
        
        // 遍历所有tensor，分配内存偏移
        for (auto &tensor : tensors)
        {
            if (tensor && allocated_tensors.find(tensor.get()) == allocated_tensors.end())
            {
                // 计算tensor需要的内存大小
                size_t tensor_size = tensor->getBytes();
                
                // 使用allocator分配内存地址偏移
                size_t offset = allocator.alloc(tensor_size);
                
                // 保存tensor和偏移的对应关系
                tensor_offsets.emplace_back(tensor, offset);
                allocated_tensors.insert(tensor.get());
            }
        }
        
        // 第二阶段：获取实际内存指针并绑定到tensor
        void *base_ptr = allocator.getPtr();
        if (base_ptr)
        {
            for (const auto &pair : tensor_offsets)
            {
                auto tensor = pair.first;
                size_t offset = pair.second;
                
                // 计算实际的内存地址
                //char* 以字节为单位进行指针算术
                void *tensor_ptr = static_cast<char*>(base_ptr) + offset;
                
                // 创建Blob对象
                Blob blob = make_ref<BlobObj>(runtime, tensor_ptr);
                
                // 绑定内存到tensor
                tensor->setDataBlob(blob);
            }
        }
        
        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

    bool GraphObj::areInverseTransposes(const TransposeObj* transpose1, const TransposeObj* transpose2) {
        if (!transpose1 || !transpose2) {
            return false;
        }
        
        auto perm1 = transpose1->getPermute();
        auto perm2 = transpose2->getPermute();
        
        if (perm1.size() != perm2.size()) {
            return false;
        }
        
        // 检查是否为互逆排列
        for (size_t i = 0; i < perm1.size(); ++i) {
            if (perm1[perm2[i]] != static_cast<int>(i)) {
                return false;
            }
        }
        
        return true;
    }
    
    bool GraphObj::areSameTransposes(const TransposeObj* transpose1, const TransposeObj* transpose2) {
        if (!transpose1 || !transpose2) {
            return false;
        }
        
        auto perm1 = transpose1->getPermute();
        auto perm2 = transpose2->getPermute();
        
        if (perm1.size() != perm2.size()) {
            return false;
        }
        
        // 检查是否为相同的排列
        for (size_t i = 0; i < perm1.size(); ++i) {
            if (perm1[i] != perm2[i]) {
                return false;
            }
        }
        
        return true;
    }
    
    bool GraphObj::isLastTwoDimsSwap(const TransposeObj* transpose) {
        if (!transpose) {
            return false;
        }
        
        auto perm = transpose->getPermute();
        if (perm.size() < 2) {
            return false;
        }
        
        // 检查最后两个维度是否被交换
        // 对于 n 维张量，最后两个维度的索引是 n-2 和 n-1
        size_t n = perm.size();
        return (perm[n-2] == static_cast<int>(n-1) && perm[n-1] == static_cast<int>(n-2));
    }
    
    void GraphObj::mergeTransposeToMatmul(const Operator& transpose, const Operator& matmul) {
        auto transpose_obj = dynamic_cast<TransposeObj*>(transpose.get());
        auto matmul_obj = dynamic_cast<MatmulObj*>(matmul.get());
        
        if (!transpose_obj || !matmul_obj) {
            return;
        }
        
        // 获取转置的输入张量
        auto transpose_inputs = transpose->getInputs();
        if (transpose_inputs.empty()) {
            return;
        }
        
        // 获取矩阵乘的输入张量
        auto matmul_inputs = matmul->getInputs();
        if (matmul_inputs.size() < 2) {
            return;
        }
        
        // 确定转置操作对应的是矩阵乘的哪个输入（A 或 B）
        auto transpose_output = transpose->getOutput();
        if (!transpose_output) {
            return;
        }
        
        bool isInputA = (transpose_output == matmul_inputs[0]);
        bool isInputB = (transpose_output == matmul_inputs[1]);
        
        if (isInputA) {
            // 转置操作在 A 输入上，设置 transA = true
            matmul_obj->setTransA(true);
            // 将矩阵乘的 A 输入改为转置的输入
            matmul->replaceInput(matmul_inputs[0], transpose_inputs[0]);
            
            // 更新张量连接关系
            transpose_inputs[0]->addTarget(matmul);
            transpose_output->removeTarget(matmul);
        } else if (isInputB) {
            // 转置操作在 B 输入上，设置 transB = true
            matmul_obj->setTransB(true);
            // 将矩阵乘的 B 输入改为转置的输入
            matmul->replaceInput(matmul_inputs[1], transpose_inputs[0]);
            
            // 更新张量连接关系
            transpose_inputs[0]->addTarget(matmul);
            transpose_output->removeTarget(matmul);
        }
    }

    void GraphObj::reconnectGraph(const Operator& op1, const Operator& op2) {
    // 参数验证
    if (!op1 || !op2) {
        return;
    }
    
    // 确保 op1 和 op2 是相邻的
    auto op1_successors = op1->getSuccessors();
    bool areAdjacent = false;
    for (const auto& succ : op1_successors) {
        if (succ == op2) {
            areAdjacent = true;
            break;
        }
    }
    
    if (!areAdjacent) {
        // 如果两个操作符不相邻，可能需要不同的处理逻辑
        return;
    }
    
    // 处理张量重新连接：将 op1 的输入直接连接到 op2 的后继节点
    auto op1_inputs = op1->getInputs();
    auto op2_outputs = op2->getOutputs();
    auto op2_successors = op2->getSuccessors();
    
    if (!op1_inputs.empty() && !op2_outputs.empty()) {
        auto input_tensor = op1_inputs[0];  // op1 的输入张量
        auto output_tensor = op2_outputs[0]; // op2 的输出张量
        
        // 将所有使用 op2 输出张量的操作符改为使用 op1 的输入张量
        auto targets = output_tensor->getTargets();
        for (const auto& target : targets) {
            if (target && target != op1 && target != op2) {
                target->replaceInput(output_tensor, input_tensor);
                input_tensor->addTarget(target);
            }
        }
    }
    
    // 获取连接信息
    auto op1_predecessors = op1->getPredecessors();
    
    // 建立新的连接：op1的前驱 -> op2的后继
    for (const auto& pred : op1_predecessors) {
        for (const auto& succ : op2_successors) {
            // 避免自环
            if (pred != succ) {
                pred->addSuccessors(succ);
                succ->addPredecessors(pred);
            }
        }
    }
    
    // 移除旧的连接
    for (const auto& pred : op1_predecessors) {
        pred->removeSuccessors(op1);
    }
    
    for (const auto& succ : op2_successors) {
        succ->removePredecessors(op2);
    }
}

void GraphObj::cleanupUnusedTensors() {
    // 清理不再被任何有效操作符使用的张量
    auto it = tensors.begin();
    while (it != tensors.end()) {
        auto tensor = *it;
        bool isUsed = false;
        
        // 检查是否有任何操作符将此张量作为输入或输出
        for (const auto& op : ops) {
            // 检查输入
            for (const auto& input : op->getInputs()) {
                if (input == tensor) {
                    isUsed = true;
                    break;
                }
            }
            if (isUsed) break;
            
            // 检查输出
            for (const auto& output : op->getOutputs()) {
                if (output == tensor) {
                    isUsed = true;
                    break;
                }
            }
            if (isUsed) break;
        }
        
        // 如果张量没有被任何操作符使用，则删除它
        if (!isUsed) {
            it = tensors.erase(it);
        } else {
            ++it;
        }
    }
}

} // namespace infini
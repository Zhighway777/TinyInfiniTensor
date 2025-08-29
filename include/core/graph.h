#pragma once
#include "core/allocator.h"
#include "core/operator.h"
#include "core/tensor.h"
#include "operators/transpose.h"
#include <algorithm>
#include <cstdint>

namespace infini
{
    // 前向声明
    class MatmulObj;

    class GraphObj : public Object
    {
    protected:
        Runtime runtime;
        TensorVec tensors;
        OpVec ops;
        Allocator allocator;

    public:
        explicit GraphObj(Runtime runtime)
            : runtime(runtime), allocator(runtime), sorted(false){};
        string toString() const override;
        Runtime getRuntime() const { return runtime; }

        Tensor addTensor(Shape dim, DataType dtype = DataType::Float32);
        Tensor addTensor(const Tensor &tensor);
        TensorVec addTensor(const TensorVec &tensors);
        void removeOperator(Operator op)
        {
            // 清理操作符与张量的连接关系
            for (auto& input : op->getInputs()) {
                if (input) {
                    input->removeTarget(op);
                }
            }
            for (auto& output : op->getOutputs()) {
                if (output) {
                    output->setSource(nullptr);
                }
            }
            
            // 清理操作符之间的连接关系
            // 从所有前驱操作符中移除对当前操作符的引用
            auto predecessors = op->getPredecessors();
            for (const auto& pred : predecessors) {
                if (pred) {
                    pred->removeSuccessors(op);
                }
            }
            
            // 从所有后继操作符中移除对当前操作符的引用
            auto successors = op->getSuccessors();
            for (const auto& succ : successors) {
                if (succ) {
                    succ->removePredecessors(op);
                }
            }
            
            // 从操作符列表中删除
            auto it = std::find(ops.begin(), ops.end(), op);
            if (it != ops.end())
                ops.erase(it);
        }

        void removeTensor(Tensor tensor)
        {
            auto it = std::find(tensors.begin(), tensors.end(), tensor);
            if (it != tensors.end())
                tensors.erase(it);
        }

        const TensorVec &getTensors() const { return tensors; }
        const OpVec &getOperators() const { return ops; }
        Tensor getTensor(int) const;

        /**
         * @brief Sort the nodes in topological order.
         * It returns true if the sorting is successful.
         * Otherwise false is returned, means that there are rings in the graph,
         * so the topological sorting fails.
         */
        bool topo_sort();

        void optimize();

        void shape_infer();

        void dataMalloc();

        /**
         * @brief Add an operator and create its outputs. Output tensor arguments
         * should be empty Refs (e.g., nullptr).
         */
        template <typename T, typename... Args>
        Ref<T> addOp(Args &&...args)
        {
            Ref<T> op = infini::make_ref<T>(this, std::forward<Args>(args)...);
            addOperatorAndConnect(op);
            return op;
        }

        /**
         * @brief Add an operator with its outputs specified.
         */
        template <typename T, typename... Args>
        Ref<T> addOpWithOutputs(Args &&...args)
        {
            Ref<T> op = infini::make_ref<T>(nullptr, std::forward<Args>(args)...);
            addOperatorAndConnect(op);
            return op;
        }

        /**
         * @brief Gets input tensors of this graph.
         */
        inline TensorVec getInputs() const
        {
            TensorVec ret;
            for (const auto &t : tensors)
                if (!t->getSource())
                    ret.emplace_back(t);
            return ret;
        }

        /**
         * @brief Gets output tensors of this graph.
         */
        inline TensorVec getOutputs() const
        {
            TensorVec ret;
            for (const auto &t : tensors)
                if (t->getTargets().empty())
                    ret.emplace_back(t);
            return ret;
        }

        bool checkValid() const;

    private:
        /**
         * @brief Add reverse connections and Op relationship in ctor.
         */
        void addOperatorAndConnect(const Operator &op);

        /**
         * @brief If the nodes is sorted in topological order.
         */
        bool sorted;

        /**
         * @brief Add check function for inverse transpose
         */
        bool areInverseTransposes(const TransposeObj *transpose1, const TransposeObj *transpose2);
        
        /**
         * @brief Add check function for same transpose
         */
        bool areSameTransposes(const TransposeObj *transpose1, const TransposeObj *transpose2);
        
        /**
         * @brief Check if transpose swaps last two dimensions
         */
        bool isLastTwoDimsSwap(const TransposeObj *transpose);
        
        /**
         * @brief Merge transpose into matmul operator
         */
        void mergeTransposeToMatmul(const Operator& transpose, const Operator& matmul);
        
        /**
         * @brief Reconnect graph after removing operators
         */
        void reconnectGraph(const Operator& op1, const Operator& op2);
        
        /**
         * @brief Clean up unused tensors
         */
        void cleanupUnusedTensors();
    };

} // namespace infini

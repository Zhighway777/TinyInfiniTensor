#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);
        
        
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================

        // 使用First Fit算法查找合适的空闲块
        for (auto it = free_blocks.begin(); it != free_blocks.end(); ++it) {
            size_t block_addr = it->first;
            size_t block_size = it->second;
            
            if (block_size >= size) {
                // 找到合适的块，进行分配
                size_t remaining_size = block_size - size;
                
                if (remaining_size > 0) {
                    // 如果剩余空间足够大，创建新的空闲块
                    free_blocks[block_addr + size] = remaining_size;
                }
                
                // 移除或更新当前块
                free_blocks.erase(it);
                
                // 更新使用统计
                used += size;
                if (used > peak) {
                    peak = used;
                }
                
                return block_addr;
            }
        }
        
        // 如果没有找到合适的空闲块，需要扩展内存
        // 这里简化处理，直接返回当前已使用的大小作为新地址
        size_t new_addr = used;
        used += size;
        if (used > peak) {
            peak = used;
        }
        
        return new_addr;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================

        // 将释放的内存块添加到空闲块映射中
        free_blocks[addr] = size;
        
        // 尝试与相邻的空闲块合并
        mergeAdjacentBlocks(addr, size);
        
        // 更新使用统计
        used -= size;
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::mergeAdjacentBlocks(size_t addr, size_t size)
    {
        // 查找前一个相邻的空闲块
        auto prev_it = free_blocks.find(addr - 1);
        if (prev_it != free_blocks.end()) {
            // 找到前一个块，检查是否真的相邻
            size_t prev_addr = prev_it->first;
            size_t prev_size = prev_it->second;
            
            if (prev_addr + prev_size == addr) {
                // 可以与前一个块合并
                size_t new_size = prev_size + size;
                free_blocks[prev_addr] = new_size;
                free_blocks.erase(addr);  // 移除当前块
                
                // 更新参数，继续检查是否可以与后一个块合并
                addr = prev_addr;
                size = new_size;
            }
        }
        
        // 查找后一个相邻的空闲块
        auto next_it = free_blocks.find(addr + size);
        if (next_it != free_blocks.end()) {
            // 找到后一个块，可以合并
            size_t next_size = next_it->second;
            size_t new_size = size + next_size;
            free_blocks[addr] = new_size;
            free_blocks.erase(addr + size);  // 移除后一个块
        }
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}

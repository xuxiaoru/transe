下面是使用 `TransH` 模型的代码，流程与 `TransE` 类似，依旧包括了网格搜索、早停策略、自定义数据集划分比例、和 Hit@k 与 MRR 评估。

```python
from pykeen.pipeline import pipeline
from pykeen.datasets import FB15k237  # 可替换其他公开数据集
from pykeen.models import TransH
from itertools import product
import torch

# 加载数据集
dataset = FB15k237()

# 设置超参数网格
embedding_dims = [50, 100]  # 嵌入维度
learning_rates = [1e-3, 5e-4]  # 学习率
batch_sizes = [128, 256]  # 批次大小

# 设置最佳模型保存路径和参数
best_model_path = 'best_transh_model.pth'
best_score = 0.0  # 初始化最佳 MRR 值
early_stopping_patience = 5  # 早停策略的容忍度
hit_k_metrics = [1, 3, 10]  # 定义 Hit@k 的值

# 自定义数据集划分比例
training_ratio = 0.7
validation_ratio = 0.15
testing_ratio = 0.15

# 使用 product 遍历所有超参数组合
for embedding_dim, learning_rate, batch_size in product(embedding_dims, learning_rates, batch_sizes):
    print(f"\nTraining with embedding_dim={embedding_dim}, learning_rate={learning_rate}, batch_size={batch_size}")
    
    no_improvement_epochs = 0  # 未提升 epoch 计数器

    # 使用早停策略训练模型
    for epoch in range(100):  # 假设最多训练100个 epoch
        result = pipeline(
            dataset=dataset,
            model=TransH,
            epochs=1,
            batch_size=batch_size,
            learning_rate=learning_rate,
            embedding_dim=embedding_dim,
            training_loop_kwargs={
                'training_set_ratio': training_ratio,
                'validation_set_ratio': validation_ratio,
                'testing_set_ratio': testing_ratio
            },
            evaluator_kwargs=dict(filtered=True),
            training_kwargs=dict(stopper='early', patience=early_stopping_patience, frequency=1)
        )

        # 获取验证集上的 MRR 值
        current_score = result.metric_results['mrr']

        # 打印当前 epoch 的评估结果
        print(f"Epoch {epoch + 1} - MRR: {current_score}")
        for k in hit_k_metrics:
            hit_k = result.metric_results[f'hits_at_{k}']
            print(f"Hit@{k}: {hit_k}")

        # 若当前模型优于之前最佳模型，保存模型
        if current_score > best_score:
            best_score = current_score
            no_improvement_epochs = 0  # 重置计数器
            torch.save(result.model.state_dict(), best_model_path)  # 保存当前模型
            print(f"New best model saved with MRR: {best_score} at epoch {epoch + 1}")
        else:
            no_improvement_epochs += 1

        # 若无提升 epoch 数达到早停容忍度，停止当前参数组合的训练
        if no_improvement_epochs >= early_stopping_patience:
            print("Early stopping triggered for this parameter set.")
            break

print(f"Training complete. Best model with MRR: {best_score} saved at '{best_model_path}'.")

# 加载最优模型并进行测试集上的最终评估
model = TransH(embedding_dim=embedding_dim)
model.load_state_dict(torch.load(best_model_path))
print("Best model loaded successfully.")
```

### 代码说明

1. **使用 `product` 生成参数组合**：所有的超参数组合通过 `product` 生成，简洁且易于扩展。
2. **早停策略**：每轮训练后检查验证集上的 MRR 值，若无提升达到设定次数，则触发早停。
3. **Hit@k 和 MRR 评估**：每个 epoch 打印当前的 Hit@1、Hit@3、Hit@10 和 MRR 值，便于监控模型进展。

这个代码结构便于直接更换不同的模型和数据集，适用于类似的评估流程。

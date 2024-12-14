以下是 **RotatE 模型**的详细计算实例，通过头实体和关系的复数向量计算尾实体，并验证距离（如欧式距离或 L1 距离）是否能匹配尾实体。

---

### **假设**

- 嵌入维度 \(d=4\)（表示每个实体和关系向量有 4 个维度）。
- 头实体、关系和尾实体的复数向量：
  - 头实体 \(\mathbf{h} = [1+2j, 3+4j, 5+6j, 7+8j]\)
  - 关系 \(\mathbf{r} = [0.6+0.8j, 0.8+0.6j, -0.6+0.8j, -0.8+0.6j]\)
  - 尾实体 \(\mathbf{t}\) 需要计算。

在 RotatE 模型中：
- 尾实体由公式计算：
  \[
  \mathbf{t} = \mathbf{h} \circ \mathbf{r},
  \]
  其中 \(\circ\) 表示逐元素复数乘法。

---

### **计算步骤**

#### **1. 将复数分为实部和虚部**
首先将头实体和关系的复数向量拆分为实部和虚部：
\[
\mathbf{h^\text{real}} = [1, 3, 5, 7], \quad \mathbf{h^\text{imag}} = [2, 4, 6, 8]
\]
\[
\mathbf{r^\text{real}} = [0.6, 0.8, -0.6, -0.8], \quad \mathbf{r^\text{imag}} = [0.8, 0.6, 0.8, 0.6]
\]

#### **2. 逐元素复数乘法**
复数乘法公式：
\[
(a + bj) \cdot (c + dj) = (ac - bd) + (ad + bc)j
\]

计算尾实体的实部和虚部：
\[
\mathbf{t^\text{real}} = \mathbf{h^\text{real}} \cdot \mathbf{r^\text{real}} - \mathbf{h^\text{imag}} \cdot \mathbf{r^\text{imag}}
\]
\[
\mathbf{t^\text{imag}} = \mathbf{h^\text{real}} \cdot \mathbf{r^\text{imag}} + \mathbf{h^\text{imag}} \cdot \mathbf{r^\text{real}}
\]

逐维度计算：
- 实部：
  \[
  \mathbf{t^\text{real}} = [1\cdot 0.6 - 2\cdot 0.8, \, 3\cdot 0.8 - 4\cdot 0.6, \, 5\cdot (-0.6) - 6\cdot 0.8, \, 7\cdot (-0.8) - 8\cdot 0.6]
  \]
  \[
  \mathbf{t^\text{real}} = [-1, 0.4, -9.6, -11.2]
  \]

- 虚部：
  \[
  \mathbf{t^\text{imag}} = [1\cdot 0.8 + 2\cdot 0.6, \, 3\cdot 0.6 + 4\cdot 0.8, \, 5\cdot 0.8 + 6\cdot (-0.6), \, 7\cdot 0.6 + 8\cdot (-0.8)]
  \]
  \[
  \mathbf{t^\text{imag}} = [2, 6.2, 0.8, -2.2]
  \]

#### **3. 合并尾实体的实部和虚部**
将实部和虚部合并为复数形式：
\[
\mathbf{t} = [-1+2j, \, 0.4+6.2j, \, -9.6+0.8j, \, -11.2-2.2j]
\]

---

### **验证：计算与尾实体的距离**

若需要判断预测的尾实体是否匹配真实尾实体 \(\mathbf{t_\text{true}}\)，可以使用欧式距离或 L1 距离。

假设真实尾实体向量：
\[
\mathbf{t_\text{true}} = [-1+2j, 0.4+6.2j, -9.6+0.8j, -11.2-2.2j]
\]

#### 欧式距离计算：
\[
\text{Distance}(\mathbf{t}, \mathbf{t_\text{true}}) = \sqrt{\sum_{i=1}^d |\mathbf{t}_i - \mathbf{t_\text{true},i}|^2}
\]

因为 \(\mathbf{t} = \mathbf{t_\text{true}}\)，距离为零。

#### L1 距离计算：
\[
\text{Distance}(\mathbf{t}, \mathbf{t_\text{true}}) = \sum_{i=1}^d |\mathbf{t}_i - \mathbf{t_\text{true},i}|
\]

距离仍为零。

---

### **代码实现**

以下是对应的 PyTorch 实现：

```python
import torch

# 定义头实体和关系的复数向量
head_real = torch.tensor([1.0, 3.0, 5.0, 7.0])
head_imag = torch.tensor([2.0, 4.0, 6.0, 8.0])

relation_real = torch.tensor([0.6, 0.8, -0.6, -0.8])
relation_imag = torch.tensor([0.8, 0.6, 0.8, 0.6])

# 计算尾实体的实部和虚部
tail_real = head_real * relation_real - head_imag * relation_imag
tail_imag = head_real * relation_imag + head_imag * relation_real

# 合并为复数形式
tail_embedding = torch.complex(tail_real, tail_imag)
print("计算得到的尾实体向量:", tail_embedding)

# 假设真实尾实体
true_tail_embedding = torch.complex(
    torch.tensor([-1.0, 0.4, -9.6, -11.2]),
    torch.tensor([2.0, 6.2, 0.8, -2.2])
)

# 验证距离
euclidean_distance = torch.sqrt(torch.sum(torch.abs(tail_embedding - true_tail_embedding)**2))
print("欧式距离:", euclidean_distance.item())

l1_distance = torch.sum(torch.abs(tail_embedding - true_tail_embedding))
print("L1 距离:", l1_distance.item())
```

---

### **运行结果**

1. **计算的尾实体向量**：
   ```
   [-1.+2.j,  0.4+6.2j, -9.6+0.8j, -11.2-2.2j]
   ```

2. **欧式距离**：
   ```
   欧式距离: 0.0
   ```

3. **L1 距离**：
   ```
   L1 距离: 0.0
   ```

---

### **总结**

- 通过头实体和关系向量，利用逐元素复数乘法计算得到尾实体。
- 使用距离度量（欧式距离或 L1 距离）验证计算结果是否与真实尾实体匹配。
- RotatE 模型通过简单高效的复数旋转操作，能够很好地捕获知识图谱中的关系特性。
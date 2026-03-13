# vk_cs 完整流程说明（中文）

本文档基于当前仓库代码，按“程序执行顺序”说明每一步做了什么，并给出对应的物理与数值公式。  
文档目标是帮助你快速定位性能瓶颈、验证数值逻辑、并做后续扩展。

## 1. 程序总体结构

当前程序由以下模块组成：

1. 主控与参数解析：`src/main.cpp`
2. 网格加载与预处理：`src/mesh_loader.cpp`
3. 子空间基底加载：`src/subspace_loader.cpp`
4. 子空间仿真主循环：`src/subspace_simulator.cpp`
5. Vulkan 计算与 shader 调度：`src/vk_two_phase.cpp`
6. Vulkan 基础封装：`src/vulkan_context.cpp`
7. 实时窗口渲染（可选）：`src/simple_renderer.cpp`（SDL3）

## 2. Mermaid 流程图

## 2.1 主控流（CPU 视角）

```mermaid
flowchart TD
    A[程序启动 main.cpp] --> B[解析命令行与选项]
    B --> C[加载 mesh 与 basis]
    C --> D[Initialize: 构建 x_rest / 质量 / CSR]
    D --> E[VkTwoPhaseOps::Initialize]
    E --> F{frame 循环}
    F --> G{substep 循环}
    G --> H[PredictState]
    H --> I{Newton 迭代}
    I --> J[ComputeGradientReducedFromQ]
    J --> K{收敛?]
    K -- 否 --> L[线性解: Direct 或 GPU-CG]
    L --> M[q += Δq]
    M --> I
    K -- 是 --> N[ReconstructAndUpdateVelocityFromQ]
    N --> O{substep 结束}
    O --> P{是否回读 / 输出 / 渲染}
    P --> Q[DownloadX]
    Q --> R{写 OBJ?}
    R -- 是 --> S[WriteObj]
    R -- 否 --> T[跳过写盘]
    S --> U{渲染?}
    T --> U
    U -- 是 --> V[RenderFrame(SDL3)]
    U -- 否 --> W[下一帧]
    V --> W
    W --> F
    F --> X[输出计时统计并退出]
```

## 2.2 GPU 调度流（Kernel / Dispatch 视角）

```mermaid
flowchart TD
    A0[子步开始] --> A1[predict_state.comp]
    A1 --> A2[reconstruct_x_from_reduced.comp]

    A2 --> B1[gradient_tet_stage.comp]
    B1 --> B2[gradient_vertex_gather.comp]
    B2 --> B3[project_stage1.comp]
    B3 --> B4[project_stage2.comp]
    B4 --> B5[得到 g_r]

    B5 --> C0{线性解路径}

    C0 -- Direct装配 --> C1[多次 ComputeHessianReduced(e_j)]
    C1 --> C2[CPU Cholesky]

    C0 -- GPU-CG --> D1[reduced_cg_init.comp]
    D1 --> D2[build_world_from_reduced.comp]
    D2 --> D3[hessp_tet_stage.comp]
    D3 --> D4[hessp_vertex_gather.comp]
    D4 --> D5[project_stage1.comp]
    D5 --> D6[project_stage2.comp]
    D6 --> D7[reduced_cg_update.comp]
    D7 --> D8{达到迭代上限或收敛?}
    D8 -- 否 --> D2
    D8 -- 是 --> E1[得到 Δq]

    C2 --> E1
    E1 --> F1[reconstruct_x_from_reduced.comp]
    F1 --> F2[update_velocity_state.comp]
    F2 --> G0[子步结束]
```

说明：

1. 图 2.2 里的 kernel 之间在代码中通过 `vkCmdPipelineBarrier` 做读写依赖同步。
2. reduced-CG 路径中，部分 dispatch 使用 indirect 参数缓冲，支持 GPU 侧早停。

## 3. 输入与参数

命令格式（典型）：

```powershell
.\build\Release\vk_cs.exe <mesh_path_or_dir> <basis_json_path> <frames> [--no-obj] [--download-only] [--render]
```

关键行为：

1. `--no-obj`：不写 OBJ；若也不渲染，通常不需要每帧回读顶点。
2. `--download-only`：每帧回读顶点，但不写 OBJ。
3. `--render`：开启实时窗口；渲染需要回读 `x`（当前实现）。
4. 若传入目录，会自动寻找 mesh 和 `_result.json`。

## 4. 网格与基底数据约束

## 4.1 网格

支持：

1. `.msh`（Gmsh ASCII）
2. `.json`（`vertices` + `tetrahedra`）

加载后会统一修正四面体方向，使体积符号一致。

## 4.2 子空间基底

从 JSON 读取：

1. `basis`（二维数组，行主序）
2. 可选 `young`、`poisson`
3. 可选 `vertex_mass`

强约束：

1. `basis.rows == 3 * num_vertices`

否则初始化失败（自由度不匹配）。

## 5. 物理参数与派生量

参数定义见 `src/sim_params.h`。核心派生公式：

$$
\mu = \frac{E}{2(1+\nu)}
$$

$$
\lambda = \frac{E\nu}{(1+\nu)(1-2\nu)}
$$

$$
\alpha = 1 + \frac{\mu}{\lambda}
$$

$$
dt = \frac{\text{time\_step}}{\text{num\_substeps}}
$$

## 6. 初始化阶段（Initialize）

执行位置：`SubspaceSimulator::Initialize`。

## 6.1 初始位形构造

对输入网格顶点做：

1. 平移到局部中心附近
2. 绕 Z 轴旋转
3. 按 `initial_height` 抬高
4. 施加可选初始拉伸

得到状态：`x_rest, x, x_n, x_star, v_n`。

## 6.2 参考四面体预计算

`BuildRestTetData` 计算：

1. 每个四面体的 $D_m^{-1}$
2. 每个四面体参考体积 $V_t$
3. 顶点 lumped mass

$$
m_i = \sum_{t \ni i}\frac{\rho V_t}{4}
$$

## 6.3 CSR 邻接构建

`BuildTetVertexCsr` 建立顶点到相邻四面体 CSR 索引（`offsets/tet_ids/local_ids`）。

## 6.4 Vulkan 初始化

`VkTwoPhaseOps::Initialize` 完成：

1. 计算设备与队列初始化
2. 静态缓冲上传：`basis/x_rest/mass/tets/DmInv/vol/CSR`
3. 动态缓冲创建：`x/x_star/x_n/v_n/p/force/result` 与 reduced/CG 缓冲
4. compute pipeline 创建与 descriptor 绑定

## 7. 能量模型与目标函数

当前实现可理解为隐式欧拉一步最小化：

$$
\Phi(x)=\frac{1}{2dt^2}\sum_i m_i\|x_i-x_i^\*\|^2
+\sum_t V_t\Psi(F_t)
+E_{\text{ground}}(x)
$$

其中：

$$
x^\* = x_n + dt\,v_n + dt^2 g
$$

$$
F = D_s D_m^{-1},\quad
D_s=[x_1-x_0,\;x_2-x_0,\;x_3-x_0]
$$

$$
\Psi(F)=\frac{\mu}{2}\|F\|_F^2 + \frac{\lambda}{2}\big(\det(F)-\alpha\big)^2
$$

$$
P=\mu F + \lambda(J-\alpha)J F^{-T},\quad J=\det(F)
$$

地面罚函数（若启用）：

$$
E_{\text{ground},i}=
\begin{cases}
\frac12 k(y_g-y_i)^2, & y_i<y_g\\
0, & y_i\ge y_g
\end{cases}
$$

## 8. 每帧与每子步流程

外层在 `Run`：

1. frame 循环
2. 每帧执行 `num_substeps` 次 `Substep`
3. 可选回读/写 OBJ/渲染

子步核心：

1. Predict：

$$
x^\* = x_n + dt\,v_n + dt^2 g,\qquad x \leftarrow x^\*
$$

2. Newton 外循环（CPU 控制，GPU 算子）：

$$
H_r \Delta q = -g_r
$$

$$
g_r = U^T g,\qquad H_r p = U^T H(U p)
$$

3. reduced 梯度：重构 `x = x_rest + Uq`，再走梯度两阶段与投影。

4. 收敛判据：

$$
\frac{\|g_r\|_2}{r} < \varepsilon_{\text{reduced}}
$$

5. 线性解：Direct 或 GPU reduced-CG。

6. 更新坐标：

$$
q \leftarrow q + \Delta q
$$

7. 子步收尾：

$$
x = x_{\text{rest}} + Uq
$$

$$
v_n = \frac{x - x_n}{dt},\qquad x_n \leftarrow x
$$

## 9. reduced-CG 更新公式

GPU-CG 每轮更新：

$$
\alpha_k=\frac{r_k^Tr_k}{p_k^T(Ap_k+\gamma p_k)}
$$

$$
x_{k+1}=x_k+\alpha_k p_k,\qquad
r_{k+1}=r_k-\alpha_k(Ap_k+\gamma p_k)
$$

$$
\beta_k=\frac{r_{k+1}^Tr_{k+1}}{r_k^Tr_k},\qquad
p_{k+1}=r_{k+1}+\beta_k p_k
$$

其中 $Ap$ 通过 GPU 链路计算：

1. `build_world_from_reduced`：$p_w = Up$
2. `hessp_tet_stage + hessp_vertex_gather`：$y_w = H p_w$
3. `project_stage1 + project_stage2`：$Ap = U^T y_w$

## 10. 输出与渲染流程

每帧子步后：

1. 若需要则 `DownloadX`
2. 若写 OBJ 则导出 `frame_*.obj`
3. 若 `--render` 则 SDL3 实时显示

当前实时渲染控制：

1. 右键拖拽旋转
2. 滚轮缩放
3. `R` 重置相机
4. `Esc` 退出渲染循环

## 11. 计时字段含义

最终输出常见字段：

1. `predict`
2. `gradient_reduced`
3. `linear_solve_total`
4. `hessian_reduced_calls`
5. `reconstruct_final`
6. `download_x`
7. `render`
8. `write_obj`

## 12. 优化导向结论

1. 纯模拟性能建议看 `--no-obj` 且不渲染。
2. 实时路径应尽量减少每帧 CPU 回读。
3. 热点通常在 reduced 梯度与 reduced 线性解。
4. 当前两阶段算子是避免原子冲突的关键设计。

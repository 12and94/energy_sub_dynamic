# GitHub 上传整理清单（建议）

这个清单用于把当前工程整理成一个可读、可复现、可持续维护的开源仓库。

## 1. 第一阶段：仓库清理

1. 确认不提交本地产物
2. 已提供 `.gitignore`，会忽略 `build/`、`*.obj`、`*.pdb`、`*.spv`、`*.vkcs_output`
3. 保留 `src/`、`shaders/`、`data/`、`libs/nlohmann/`、`CMakeLists.txt`、`README.md`

## 2. 第二阶段：数据策略

1. `data/example_1` 建议保留（轻量、可快速验证）
2. `data/example_3` 体积较大（`_result.json` 约 40MB），当前已在 `.gitignore` 默认排除
3. 如果你希望公开该数据，可移除 `.gitignore` 中 `/data/example_3/`
4. 或改为 Release 附件 / 网盘下载链接
5. 或使用 Git LFS 管理大文件

## 3. 第三阶段：文档一致性

1. `README.md` 作为首页，讲清楚：
2. 这是什么项目
3. 独特技术点（two-phase、reduced GPU 算子、GPU-CG）
4. 如何构建和运行
5. 建议修复历史中文文档编码（例如 `PIPELINE_CN.md` 若出现乱码，统一转 UTF-8）

## 4. 第四阶段：提交与推送

首次初始化仓库示例：

```powershell
git init
git add .
git commit -m "init: vulkan compute subspace fem prototype"
git branch -M main
git remote add origin <your_github_repo_url>
git push -u origin main
```

## 5. 推荐首批标签（Topics）

在 GitHub 仓库设置中添加：

- `vulkan`
- `compute-shader`
- `fem`
- `physics-simulation`
- `subspace`
- `gpu-computing`

## 6. 首版发布建议

1. 打一个 `v0.1.0` tag，说明当前是原型版本
2. 在 Release 里给出：
3. 支持平台（当前建议 Windows）
4. 已验证示例数据
5. 已知限制（渲染非重点、参数配置仍偏硬编码）

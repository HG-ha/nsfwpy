# 发布指南 / Release Guide

## PyPI 发布触发方式

### 方式 1：使用 Git Tags（推荐）⭐

这是最标准和推荐的方式：

```bash
# 1. 确保所有更改已提交
git add .
git commit -m "准备发布 v1.0.0"

# 2. 更新 setup.py 中的版本号
# 编辑 setup.py，修改 version='1.0.0'

# 3. 创建并推送标签
git tag v1.0.0
git push origin main
git push origin v1.0.0

# 或者一次性推送所有标签
git push origin main --tags
```

**标签格式要求：** `v*.*.*` (例如 v1.0.0, v2.1.3)

### 方式 2：手动触发

在 GitHub 仓库页面操作：

1. 进入仓库页面
2. 点击 **Actions** 标签
3. 选择 **Publish to PyPI** 工作流
4. 点击 **Run workflow** 按钮
5. 选择分支（通常是 main）
6. 点击 **Run workflow** 确认

### 方式 3：创建 GitHub Release（推荐用于正式版本）

1. 在 GitHub 仓库页面点击 **Releases**
2. 点击 **Create a new release**
3. 选择或创建标签（如 `v1.0.0`）
4. 填写 Release 标题和描述
5. 点击 **Publish release**

这会自动触发标签推送，进而触发 PyPI 发布。

## 发布前检查清单

- [ ] 更新 `setup.py` 中的版本号
- [ ] 更新 `pyproject.toml` 中的版本号（如果有）
- [ ] 更新 `README.md` 和 `README_EN.md`
- [ ] 测试所有功能正常工作
- [ ] 检查 `requirements.txt` 依赖是否正确
- [ ] 运行测试（如果有）
- [ ] 更新 CHANGELOG（如果有）

## 版本号规范

遵循语义化版本 (Semantic Versioning):

- **主版本号 (Major)**: 不兼容的 API 修改
- **次版本号 (Minor)**: 向下兼容的功能性新增
- **修订号 (Patch)**: 向下兼容的问题修正

示例：
- `v1.0.0` - 首个稳定版本
- `v1.1.0` - 添加新功能
- `v1.1.1` - 修复 bug
- `v2.0.0` - 重大更新，可能有破坏性变更

## 注意事项

1. **PyPI API Token**: 确保在 GitHub 仓库设置中配置了 `PYPI_API_TOKEN` secret
   - 位置: Settings → Secrets and variables → Actions → New repository secret
   - 名称: `PYPI_API_TOKEN`
   - 值: 从 PyPI 获取的 API token

2. **版本唯一性**: PyPI 不允许重复的版本号，发布后无法删除

3. **测试发布**: 如需测试发布流程，可以使用 TestPyPI:
   ```bash
   # 发布到 TestPyPI
   python -m build
   twine upload --repository testpypi dist/*
   ```

## 快速发布命令

```bash
# 一键发布脚本
VERSION="1.0.0"
git add .
git commit -m "Release v${VERSION}"
git tag v${VERSION}
git push origin main --tags
```

## 查看发布状态

- GitHub Actions: https://github.com/HG-ha/nsfwpy/actions
- PyPI 项目页: https://pypi.org/project/nsfwpy/


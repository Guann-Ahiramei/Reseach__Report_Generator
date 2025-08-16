# Industry Reporter 2 - End-to-End Testing Suite

这个测试套件为Industry Reporter 2的所有核心功能提供全面的端到端测试。

## 测试覆盖范围

### 🗄️ Redis服务测试 (`test_redis_service.py`)
- 连接和基本操作
- JSON序列化/反序列化
- 批量操作
- 模式匹配和键管理
- 命名空间操作
- 内存使用和性能监控
- 健康检查和错误处理

### 🔍 FAISS服务测试 (`test_faiss_service.py`)
- 服务初始化和配置
- 文档处理和分块
- 相似性搜索
- 过滤搜索
- MMR (Maximum Marginal Relevance) 搜索
- 向量搜索
- 索引持久化和优化
- 健康检查和错误处理

### 📄 文档加载器测试 (`test_document_loader.py`)
- 加载器初始化
- 单文件加载 (TXT, JSON, CSV, HTML, Python等)
- 批量文档加载
- 递归目录扫描
- 文件大小过滤
- 元数据提取
- 内容过滤
- 编码检测
- 文件支持检查
- 错误处理

### 🔄 多检索器系统测试 (`test_multi_retriever_system.py`)
- 检索器工厂模式
- 各个检索器独立功能
- 并行搜索执行
- 结果合并和排序
- 域名过滤
- 搜索质量指标
- 错误处理
- 端到端工作流

### 🧠 Skills模块测试 (`test_skills_modules.py`)
- ContextManager (上下文管理)
- ResearchConductor (研究执行)
- ReportGenerator (报告生成)
- FAISSManager (向量操作)
- HybridSearcher (混合搜索)
- 模块间集成
- 错误处理和恢复能力

## 快速开始

### 前置条件

1. **Redis服务器**
   ```bash
   # 启动Redis (确保在端口6379运行)
   redis-server
   ```

2. **Python依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **环境变量**
   ```bash
   # 设置必要的API密钥
   export OPENAI_API_KEY="your_openai_api_key"
   export TAVILY_API_KEY="your_tavily_api_key"
   ```

### 运行测试

#### 运行所有测试
```bash
cd backend/tests
python run_all_tests.py --verbose
```

#### 运行单个测试套件
```bash
# Redis服务测试
python test_redis_service.py

# FAISS服务测试
python test_faiss_service.py

# 文档加载器测试
python test_document_loader.py

# 多检索器系统测试
python test_multi_retriever_system.py

# Skills模块测试
python test_skills_modules.py
```

#### 仅运行健康检查
```bash
python run_all_tests.py --health-check-only
```

#### 保存测试结果
```bash
python run_all_tests.py --save-results --verbose
```

#### 在首次失败时停止
```bash
python run_all_tests.py --stop-on-failure --verbose
```

## 测试命令选项

### `run_all_tests.py` 选项
- `--verbose` / `-v`: 启用详细输出
- `--stop-on-failure` / `-s`: 在首次关键失败时停止
- `--health-check-only` / `-h`: 仅运行健康检查
- `--save-results` / `-r`: 将测试结果保存到JSON文件

## 测试环境说明

### Redis数据库分配
- 测试使用不同的Redis数据库编号以避免冲突：
  - Database 12: 健康检查
  - Database 13: Skills模块测试
  - Database 14: 多检索器系统测试
  - Database 15: Redis服务测试

### 临时文件和目录
- 所有测试使用临时目录和文件
- 测试完成后自动清理
- FAISS索引文件在测试目录中创建和删除

### 测试数据
- 每个测试套件创建自己的测试数据
- 包含多种格式的文档和多种类型的查询
- 测试数据设计用于验证各种功能场景

## 测试结果解读

### 成功指标
- ✅ **PASSED**: 所有测试用例通过
- 📊 **成功率**: 应该达到100%
- ⏱️ **性能**: 响应时间在合理范围内
- 🔍 **覆盖率**: 所有核心功能都被测试

### 失败排查
1. **Redis连接失败**
   - 检查Redis服务器是否运行
   - 验证连接URL和端口

2. **FAISS初始化失败**
   - 检查OpenAI API密钥是否正确
   - 验证网络连接

3. **文档加载失败**
   - 检查文件权限
   - 验证支持的文件格式

4. **Skills模块失败**
   - 检查所有依赖服务状态
   - 验证API配置

## 持续集成

这些测试设计用于：
- 开发过程中的本地验证
- CI/CD管道中的自动化测试
- 部署前的系统验证
- 性能回归检测

## 测试最佳实践

1. **运行频率**
   - 每次代码更改后运行相关测试
   - 部署前运行完整测试套件
   - 定期运行健康检查

2. **故障处理**
   - 首先运行健康检查确定基础服务状态
   - 查看详细错误日志和堆栈跟踪
   - 逐个运行单独的测试套件进行调试

3. **性能监控**
   - 关注测试执行时间趋势
   - 监控内存和资源使用
   - 检查搜索质量指标

## 扩展测试

要添加新的测试：

1. 在相应的测试文件中添加新的测试方法
2. 遵循现有的命名约定 (`test_*`)
3. 使用异步测试函数 (`async def`)
4. 包含适当的断言和错误处理
5. 更新 `run_all_tests.py` 如果添加新的测试套件

## 支持和问题

如果遇到测试问题：
1. 检查前置条件是否满足
2. 查看测试输出中的详细错误信息
3. 运行单个测试套件进行隔离调试
4. 检查Redis和API服务状态

---

**注意**: 这些测试需要有效的OpenAI和Tavily API密钥才能完全运行。没有这些密钥，某些测试可能会失败或跳过。
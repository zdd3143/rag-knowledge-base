# 从零实现的 RAG 检索系统

> 不依赖任何 RAG 框架、从零手写的检索增强生成系统。
> 面向石油石化领域文档问答。

## 这是什么

输入一批企业年报 PDF，能对里面的内容提问并给出有引用来源的答案。

**核心思路**：把文档切成小块 → 每块转成向量 → 提问时找最相似的几块 → 交给大模型作答。

## 为什么不用 LangChain

市面上 99% 的 RAG 项目是 `LangChain + Chroma + 一句话调用`，
看不出任何工程细节。这个项目**刻意不用框架**，每个环节自己实现，
并配一套评估体系来量化每一步的效果。

## 技术栈

Python · sentence-transformers (BGE-M3) · PyMuPDF · numpy

## 进度

- [x] 阶段 1：文本向量化 + 余弦相似度检索
- [x] 阶段 2：PDF 解析 + 分块（验证了「整份文档当向量会失效」）
- [ ] 阶段 3：索引持久化 + 检索加速
- [ ] 阶段 4：BM25 关键词检索
- [ ] 阶段 5：混合检索 + 重排
- [ ] 阶段 6：接入大模型生成 + 引用溯源
- [ ] 阶段 7：评估体系（量化指标）
- [ ] 阶段 8：API 服务 + 演示界面

## 目录结构
src/ stage1.py 最小 RAG：向量化 + 检索 stage2_extract.py PDF -> 文本 stage2_chunk.py 文本 -> 分块
## 运行

```bash
pip install -r requirements.txt
python src/stage3_build_index.py   # 建索引（跑一次）
python src/stage3_search.py        # 检索


写完提交：

```powershell
git add .
git commit -m "补 README：说明项目定位、动机与进度"
git push

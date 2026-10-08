 **Docling**官方目前最简单的安装方式就是 `pip install docling`，Windows、Linux、macOS 都支持。([DocLing](https://docling-project.github.io/docling/getting_started/installation/?utm_source=chatgpt.com))

## 1. 最快安装

你的 Windows 环境下，建议新建一个独立环境。

### Conda

```bash
conda create -n docling python=3.11 -y
conda activate docling
pip install -U pip
pip install docling
```

然后检查：

```bash
docling --help
```

能看到帮助信息，就说明基本安装成功。

------

# 2. 最简单的使用方法

假设你有：

```text
D:\RAG\test.pdf
```

直接：

```bash
docling "D:\RAG\test.pdf"
```

Docling CLI 默认会进行转换，并输出 Markdown；官方 Quickstart 也是这种使用方式。([DocLing](https://docling-project.github.io/docling/getting_started/quickstart/?utm_source=chatgpt.com))

不过你做 RAG，我更建议直接输出：

```bash
docling "D:\RAG\test.pdf" --to json --to md
```

这样同时得到：

```text
JSON
Markdown
```

其中：

- **JSON**：作为 RAG 后续处理的核心数据
- **Markdown**：方便人工检查解析结果

Docling CLI 当前支持 `md / json / yaml / html / text / chunks / latex` 等输出格式。([DocLing](https://docling-project.github.io/docling/reference/cli/?utm_source=chatgpt.com))

------

# 3. 批量处理整个文件夹

例如：

```text
D:\RAG\documents\
    ├── 客服手册.pdf
    ├── 电子器件手册.pdf
    ├── 论文.pdf
    ├── 小说.epub
    ├── 花名册.xlsx
    └── 产品说明书.docx
```

可以直接：

```bash
docling "D:\RAG\documents" --to json --to md
```

Docling 的 CLI 支持本地文件和目录路径，并且默认接受支持的多种格式。([DocLing](https://docling-project.github.io/docling/reference/cli/?utm_source=chatgpt.com))

------

# 4. 用 Python 更适合你后面的 RAG

真正做 RAG，我建议不要长期依赖命令行，而是直接调用 Python API。

新建：

```text
test_docling.py
```

写：

```python
from docling.document_converter import DocumentConverter

file_path = r"D:\RAG\test.pdf"

converter = DocumentConverter()

result = converter.convert(file_path)

doc = result.document

# 输出 Markdown
markdown = doc.export_to_markdown()

with open("output.md", "w", encoding="utf-8") as f:
    f.write(markdown)

print("解析完成！")
print(markdown[:2000])
```

运行：

```bash
python test_docling.py
```

官方 Quickstart 的核心 API 就是：

```python
converter = DocumentConverter()
doc = converter.convert(source).document
```

然后可以：

```python
doc.export_to_markdown()
```

导出 Markdown。([DocLing](https://docling-project.github.io/docling/getting_started/quickstart/?utm_source=chatgpt.com))

------

# 5. 先做一个“万能测试程序”

我更推荐你直接用下面这个，因为你的目标是做 RAG，不应该只测试 PDF。

```python
from pathlib import Path
from docling.document_converter import DocumentConverter

# 输入文件
input_file = Path(r"D:\RAG\test.pdf")

# 输出目录
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# 创建转换器
converter = DocumentConverter()

print(f"正在解析：{input_file}")

# 解析
result = converter.convert(input_file)

doc = result.document

# =========================
# 1. 保存 Markdown
# =========================
markdown = doc.export_to_markdown()

md_file = output_dir / f"{input_file.stem}.md"

with open(md_file, "w", encoding="utf-8") as f:
    f.write(markdown)

# =========================
# 2. 保存文本
# =========================
text = doc.export_to_text()

txt_file = output_dir / f"{input_file.stem}.txt"

with open(txt_file, "w", encoding="utf-8") as f:
    f.write(text)

print("解析完成")
print(f"Markdown：{md_file}")
print(f"Text：{txt_file}")
```

这样你就可以开始测试：

```text
PDF
DOCX
PPTX
XLSX
EPUB
HTML
```

而不用为每种文件写一套解析代码。

Docling 的定位就是把不同格式统一转换成 `DoclingDocument`，这也是它特别适合你这种通用 RAG 的原因。([DocLing](https://docling-project.github.io/docling/?utm_source=chatgpt.com))

------

# 6. 如果你要做 RAG，建议直接试 Chunk

这是 Docling 非常值得利用的一个功能。

它的 CLI 可以直接：

```bash
docling "D:\RAG\test.pdf" --to chunks
```

而且目前支持：

```text
hybrid
hierarchical
```

两种 chunk 类型。([DocLing](https://docling-project.github.io/docling/reference/cli/?utm_source=chatgpt.com))

例如：

```bash
docling "D:\RAG\test.pdf" --to chunks --chunks-type hierarchical
```

或者：

```bash
docling "D:\RAG\test.pdf" --to chunks --chunks-type hybrid
```

不过这里我建议你：

> **第一阶段不要直接使用 Docling 自动 Chunk。**

先把：

```text
原始文件
   ↓
Docling
   ↓
DoclingDocument / JSON
```

这一层做好。

然后你自己根据：

```text
章节
标题
段落
表格
页面
父子关系
```

做 RAG Chunk。

这样后面你才能控制检索效果。

------

# 7. 如果你有 NVIDIA GPU

你之前提到过 RTX 4060，这里有一个好消息：

Docling 可以使用 NVIDIA GPU；官方目前提供 CUDA 版 PyTorch 的安装方式，例如 CUDA 12.8：

```bash
pip uninstall torch torchaudio -y

pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
```

然后：

```bash
pip install docling
```

官方说明，安装 CUDA-enabled PyTorch 后，Docling 在基本使用中可以自动检测并使用 RTX GPU。([DocLing](https://docling-project.github.io/docling/getting_started/rtx/?utm_source=chatgpt.com))

检查 GPU：

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

如果看到：

```text
CUDA: True
GPU: NVIDIA GeForce RTX 4060
```

就可以了。

------

# 8. 扫描 PDF 要另外注意

普通 PDF：

```text
文字型 PDF
↓
Docling
```

通常直接就可以。

但如果 PDF 是：

```text
扫描件
图片 PDF
拍照 PDF
```

需要 OCR。

Docling 当前支持多种 OCR 引擎，包括 EasyOCR、Tesseract、RapidOCR 等。([DocLing](https://docling-project.github.io/docling/getting_started/installation/?utm_source=chatgpt.com))

例如可以安装：

```bash
pip install "docling[rapidocr]"
```

然后根据 OCR pipeline 配置使用。

**一开始不要装一大堆 OCR。**

你先拿真实文件测试：

```text
1 个普通 PDF
1 个扫描 PDF
1 个论文 PDF
1 个 DOCX
1 个 XLSX
1 个 EPUB
```

再决定是否额外安装 OCR。

------

# 9. 最推荐你的安装方式

结合你现在要做的 RAG，我建议直接：

```bash
conda create -n rag-docling python=3.11 -y
conda activate rag-docling

python -m pip install -U pip
pip install docling
```

然后：

```bash
docling "D:\RAG\test.pdf" --to json --to md
```

跑通之后再写 Python：

```text
文件
 ↓
Docling
 ↓
JSON
 ↓
结构检查
 ↓
Chunk
 ↓
Embedding
 ↓
Qdrant
 ↓
Reranker
 ↓
LLM
```

**不要一开始就装 `docling[vlm]`。**普通文档解析先把标准 pipeline 跑通；只有遇到复杂视觉文档、图表或扫描件解析效果不够时，再引入 VLM/OCR。Docling 官方也把 VLM、OCR 等作为额外能力提供。([DocLing](https://docling-project.github.io/docling/getting_started/installation/?utm_source=chatgpt.com))

### 如果你的目标是“最快看到解析效果”

实际上只需要三条命令：

```bash
conda create -n rag-docling python=3.11 -y
conda activate rag-docling
pip install docling
```

然后：

```bash
docling "你的文件.pdf" --to json --to md
```

就已经可以开始了。
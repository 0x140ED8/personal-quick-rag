from docling.document_converter import DocumentConverter

file_path = r"STM32F103x8.pdf"

converter = DocumentConverter()

result = converter.convert(file_path)

doc = result.document

# 输出 Markdown
markdown = doc.export_to_markdown()

with open("STM32F103x8.md", "w", encoding="utf-8") as f:
    f.write(markdown)

print("解析完成！")
print(markdown[:2000])
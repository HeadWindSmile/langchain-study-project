# 负责加载文档

from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document


# def load_documents(file_path: str) -> list[Document]:
#     path = Path(file_path)
#
#     if not path.exists():
#         raise FileNotFoundError(f"文件不存在: {file_path}")
#
#     suffix = path.suffix.lower()
#
#     if suffix == ".txt":
#         loader = TextLoader(str(path), encoding="utf-8")
#         return loader.load()
#
#     if suffix == ".pdf":
#         loader = PyPDFLoader(str(path))
#         return loader.load()
#
#     raise ValueError(f"暂不支持的文件类型: {suffix}")


from pathlib import Path

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document


SUPPORTED_SUFFIXES = {".txt", ".pdf"}


def load_document(file_path: str) -> list[Document]:
    """
    加载单个文档文件，目前支持 txt 和 pdf。
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    if not path.is_file():
        raise ValueError(f"不是文件: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
        return loader.load()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
        return loader.load()

    raise ValueError(f"暂不支持的文件类型: {suffix}")


def load_documents_from_dir(dir_path: str) -> list[Document]:
    """
    加载目录下所有支持的文档文件。
    当前支持：txt、pdf。
    """
    directory = Path(dir_path)

    if not directory.exists():
        raise FileNotFoundError(f"目录不存在: {dir_path}")

    if not directory.is_dir():
        raise ValueError(f"不是目录: {dir_path}")

    all_docs: list[Document] = []

    for file_path in sorted(directory.rglob("*")):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()

        if suffix not in SUPPORTED_SUFFIXES:
            print(f"跳过不支持的文件: {file_path}")
            continue

        try:
            docs = load_document(str(file_path))
            all_docs.extend(docs)
            print(f"已加载文件: {file_path}, Document 数量: {len(docs)}")
        except Exception as e:
            print(f"加载失败: {file_path}, 错误: {e}")

    if not all_docs:
        raise ValueError(f"目录中没有成功加载任何支持的文档: {dir_path}")

    return all_docs
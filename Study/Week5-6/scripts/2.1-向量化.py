import argparse
import os
import glob
from tqdm import tqdm  # 进度条
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def main():
    parser = argparse.ArgumentParser(description="Build Vector DB from Folder")
    parser.add_argument("--data_dir", type=str, required=True, help="存放PDF的文件夹路径")
    parser.add_argument("--save_dir", type=str, default="vector_db/cress_index", help="向量库保存路径")
    parser.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    # 1. 扫描所有 PDF
    pdf_files = glob.glob(os.path.join(args.data_dir, "*.pdf"))
    if not pdf_files:
        print(f"错误：在 {args.data_dir} 没找到 PDF 文件！")
        return
    print(f"发现 {len(pdf_files)} 篇文献，准备处理...")

    # 2. 逐个读取并切分 (使用进度条)
    all_splits = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)

    print("正在读取并切分文档...")
    for pdf_file in tqdm(pdf_files):
        try:
            loader = PyPDFLoader(pdf_file)
            docs = loader.load()
            # 这里可以加个元数据处理，比如文件名作为 source
            for doc in docs:
                doc.metadata['source'] = os.path.basename(pdf_file)
            
            splits = text_splitter.split_documents(docs)
            all_splits.extend(splits)
        except Exception as e:
            print(f"\n跳过损坏文件 {pdf_file}: {e}")

    print(f"总共切分出 {len(all_splits)} 个知识片段。")

    # 3. 向量化并保存 (最耗时的一步)
    print(f"正在计算向量 (使用 GPU)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=args.embed_model, 
        model_kwargs={'device': 'cuda'} # 确保用 GPU
    )
    
    # 建立索引
    vector_store = FAISS.from_documents(all_splits, embeddings)
    
    # 4. 保存到硬盘
    print(f"正在保存索引到 {args.save_dir} ...")
    vector_store.save_local(args.save_dir)
    print("✅ 建库完成！以后直接加载这个文件夹即可。")

if __name__ == "__main__":
    main()
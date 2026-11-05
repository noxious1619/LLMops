import re

with open("multi_doc_chat/src/document_ingestion/data_ingestion.py", "r", encoding="utf-8") as f:
    content = f.read()

# Pattern for def _split
pattern_split = r'def _split\(self, docs, chunk_size=500, chunk_overlap=100\):.*?return chunks'
replacement_split = '''    def _split(self, docs: List[Document], chunk_size: int = 500, chunk_overlap: int = 100) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(docs)

        for index, chunk in enumerate(chunks):
            c_text = chunk.page_content
            content_hash = hashlib.sha256(c_text.encode("utf-8")).hexdigest()
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", "unknown")
            unique_string = f"{source}_{page}_{index}_{content_hash}"
            chunk_id = hashlib.sha256(unique_string.encode("utf-8")).hexdigest()
            chunk.metadata["chunk_id"] = chunk_id
            chunk.metadata["content_hash"] = content_hash

        log.info(
            "Documents split",
            chunks=len(chunks),
            chunk_size=chunk_size,
            overlap=chunk_overlap
        )
        return chunks'''

new_content, n1 = re.subn(pattern_split, replacement_split, content, flags=re.DOTALL)
print(f"Substitutions for _split: {n1}")

pattern_loc = r'(self\.vs = FAISS\.from_texts\(texts=texts, embedding=self\.emb, metadatas=metadatas or \[\]\)\s+self\.vs\.save_local\(str\(self\.index_dir\)\)\s+return self\.vs)'
replacement_loc = '''self.vs = FAISS.from_texts(texts=texts, embedding=self.emb, metadatas=metadatas or [])
        for t, md in zip(texts, metadatas or [{}] * len(texts)):
            key = self._fingerprint(t, md)
            self._meta["rows"][key] = True
        self.vs.save_local(str(self.index_dir))
        self._save_meta()
        return self.vs'''

new_content, n2 = re.subn(pattern_loc, replacement_loc, new_content)
print(f"Substitutions for load_or_create: {n2}")

if n1 == 1 and n2 == 1:
    with open("multi_doc_chat/src/document_ingestion/data_ingestion.py", "w", encoding="utf-8") as f:
        f.write(new_content)
    print("data_ingestion.py successfully updated!")
else:
    print("Warning: regex pattern did not match expected count.")


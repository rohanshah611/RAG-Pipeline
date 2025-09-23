from unstructured.partition.pdf import partition_pdf
from langchain_openai import OpenAIEmbeddings

from typing import Literal

class PDFProcessor:
    def __init__(self, file_path: str, chunking_strategy: Literal["by_title", "basic"], maximum_chunk_characters: int,  combine_text_under_n_chars: int, new_after_n_chars: int):
        self.file_path = file_path
        self.chunking_strategy = chunking_strategy
        self.maximum_chunk_characters = maximum_chunk_characters
        self.combine_text_under_n_chars = combine_text_under_n_chars
        self.new_after_n_chars = new_after_n_chars

    def unstructured_chunks(self):
        return partition_pdf(
            filename=self.file_path,
            infer_table_structure=True,
            strategy="hi_res",
            chunking_strategy=self.chunking_strategy,
            max_characters=self.maximum_chunk_characters,              
            combine_text_under_n_chars=self.combine_text_under_n_chars,
            new_after_n_chars=self.new_after_n_chars,

        )

    def split_elements_from_chunks(self, chunk):
        """
        Extract text/table content from a CompositeElement chunk 
        and combine into one string.
        """
        if "CompositeElement" not in str(type(chunk)):
            return ""

        chunk_elements = chunk.metadata.orig_elements
        combined_content = ""

        for element in chunk_elements:
            if "Text" in str(type(element)):
                combined_content += element.text + "\n"
            elif "Table" in str(type(element)):
                combined_content += element.metadata.text_as_html + "\n"
            # elif "Image" in str(type(element)):
            #     combined_content += "[IMAGE]\n"

        return combined_content

from pymupdf import open
import re
from typing import List
from pathlib import Path
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__) , "..")))

def extract_paragraphs(pdf_path : str | Path ) -> List[str]:
    """This function takes a path of pdf and returns a List of paragraphs

    Args:
        pdf_path (str | Path): _description_

    Returns:
        List[str]: _description_
    """
    paragraphs = []
    
    doc = open(pdf_path)
    
    for page in doc:
        
        blocks = page.get_text("blocks")
        
        blocks = sorted(blocks, key=lambda b:(b[1], b[0]))
        
        for block in blocks :
            text = block[4]
        
            text = re.sub(r"\r+" , " " , text).strip() 
            
            if text :
                paragraphs.append(text)
            
    return paragraphs
#/usr/bin/env python

#1
###使用模型 掩码填充（mask filling 
from transformers import pipeline
camembert_fill_mask =pipeline("fill-mask",model="camembert-base")
results = camembert_fill_mask("Le camembert est <mask> :)")
print(results)
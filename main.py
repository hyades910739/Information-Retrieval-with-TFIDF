#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###### read data
import os
from VectorSpace import VectorSpace

documents = {}

print("loading documents...")
for filename in os.listdir("documents/"):
    with open( ("documents/"+filename),"r") as file:
        document = (file.read())
        documents[filename.split(".")[0]] = document
        file.close()

            
vectorSpace = VectorSpace(documents)    
res = vectorSpace.search("drill wood sharp","TFCOS",True)
res = vectorSpace.search("drill wood sharp","TFED",True)
res = vectorSpace.search("drill wood sharp","TFIDFCOS",True)
res = vectorSpace.search("drill wood sharp","TFIDFED",True)
res = vectorSpace.search("drill wood sharp","RelevFeedback",True)





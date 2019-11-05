# -*- coding: utf-8 -*-
from models.model import Model

class ARCMA_Model(Model):
    
    file = "arcma.db"
    table_name = "arcma"
    
    def __init__(self):
        super().__init__(self.file, self.table_name)
        
        

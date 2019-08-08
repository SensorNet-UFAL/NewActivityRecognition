# -*- coding: utf-8 -*-
from models.model import Model

class HMP_Model(Model):
    
    file = "hmp.db"
    table_name = "hmp"
    
    def __init__(self):
        super().__init__(self.file, self.table_name)
        
        


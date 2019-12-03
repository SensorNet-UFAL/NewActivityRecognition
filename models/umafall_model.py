# -*- coding: utf-8 -*-
from models.model import Model

class UMAFALL_Model(Model):
    
    file = "umafall.db"
    table_name = "umafall"
    
    def __init__(self):
        super().__init__(self.file, self.table_name)


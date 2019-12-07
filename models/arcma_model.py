# -*- coding: utf-8 -*-
from models.model import Model

class ARCMA_Model(Model):
    
    file = "arcma.db"
    table_name = "arcma"
    activity_dict = {1:"Working at Computer", 2:"Standing Up", 3:"Standing", 4:"Walking", 5:"Up\Down Stairs", 6:"Walking and Talking", 7:"Talking while Standing"}
    
    def __init__(self):
        super().__init__(self.file, self.table_name)
        
        

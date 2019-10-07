# -*- coding: utf-8 -*-
from utils.project import Project
import dill
import pickle

class save(object):
    
    path = Project.project_root+"\\workspace\\"
    #load the session again:
    def load_session(self, filename = 'globalsave.pkl'):
        dill.load_session(self.path+filename)
    #Save the session
    def save_session(self, filename = 'globalsave.pkl'):
        dill.dump_session(self.path+filename)
    
    def save_var(self, var, filename):
        with open(self.path+filename, 'wb') as f:
            pickle.dump(var, f)
    def load_var(self, filename):
        with open(self.path+filename, 'rb') as f:
            r = pickle.load(f)
        return r
    

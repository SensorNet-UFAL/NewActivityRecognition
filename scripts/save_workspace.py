# -*- coding: utf-8 -*-
from utils.project import Project
import dill
filename = Project.project_root+"\\workspace\\"+'globalsave.pkl'

#Save the session
dill.dump_session(filename)


#load the session again:
dill.load_session(filename)

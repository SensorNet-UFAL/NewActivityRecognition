import sys
from utils.project import Project, slash

exec(open(Project.project_root+"{}scripts{}arcma{}arcma_select_best_algorithm.py".format(slash, slash, slash)).read())
exec(open(Project.project_root+"{}scripts{}hmp{}hmp_select_best_algorithm.py".format(slash, slash, slash)).read())
exec(open(Project.project_root+"{}scripts{}umafall{}umafall_select_best_algorithm.py".format(slash, slash, slash)).read())

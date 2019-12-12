# -*- coding: utf-8 -*-
import sys
from utils.project import slash

exec(open("scripts{}arcma{}arcma_select_best_window.py".format(slash, slash)).read())
exec(open("scripts{}hmp{}hmp_select_best_window.py".format(slash, slash)).read())
exec(open("scripts{}umafall{}umafall_select_best_window.py".format(slash, slash)).read())


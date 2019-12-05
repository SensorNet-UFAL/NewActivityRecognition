# -*- coding: utf-8 -*-
import sys
from utils.project import slash

exec(open("scripts{}arcma{}arcma_get_all_relevant_features_each_person.py".format(slash, slash)).read())
exec(open("scripts{}hmp{}hmp_get_all_relevant_features_each_person.py".format(slash, slash)).read())
exec(open("scripts{}umafall{}umafall_get_all_relevant_features_each_person.py".format(slash, slash)).read())


# -*- coding: utf-8 -*-

class Debug:
    DEBUG = 0
    def print_debug(msg):
        if Debug.DEBUG:
            print("----------------------------------------------")
            print("Debug: {}".format(msg))
            print("----------------------------------------------")

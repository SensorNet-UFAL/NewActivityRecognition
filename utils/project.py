import datetime

slash = "/"
#slash = "\\"

class Project(object):
    project_root = "C:{}Users{}WYLKEN-DNIT{}Desktop{}NewActivityRecognition".format(slash, slash, slash, slash)
    #project_root = "{}home{}wylken.machado@laccan.net{}NewActivityRecognition".format(slash, slash, slash)
    #project_root = "C:{}Users{}wylken.machado.INTRA{}Desktop{}NewActivityRecognition".format(slash, slash, slash, slash)
    
    def log(self, str, file="log.log"):
        now = datetime.datetime.now()
        f = open("{}{}workspace{}{}".format(self.project_root, slash, slash, file), "a")
        f.write("{}/{}/{} - {}:{}:{} -> {}\r".format(now.year, now.month, now.day, now.hour, now.minute, now.second, str))
        f.close()
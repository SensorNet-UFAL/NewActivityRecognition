import datetime

class Project(object):
    #project_root = "C:\\Users\\WYLKEN-DNIT\\Documents\\NewActivityRecognition"
    #project_root = "C:\\Users\\wylken.machado.INTRA\\Desktop\\NewActivityRecognition"
    project_root = "/home/wylken.machado@laccan.net/NewActivityRecognition"
    
    def log(self, str):
        now = datetime.datetime.now()
        f = open("{}\\workspace\\log.log".format(self.project_root), "a")
        f.write("{}/{}/{} - {}:{}:{} -> {}\r".format(now.year, now.month, now.day, now.hour, now.minute, now.second, str))
        f.close()
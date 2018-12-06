# -*- coding: utf-8 -*-
import tornado.httpserver
import tornado.websocket
import tornado.ioloop
import tornado.web
import socket
import json
from pathlib import Path
from os.path import isfile
import os
import time
from ml_process import classify
import collections
import numpy as np
import sys



class WSHandler(tornado.websocket.WebSocketHandler):
    def open(self):
        print('new connection')

    def on_message(self, message):
        num = 0
        data=json.loads(message.decode('utf-8'))
        #print(type(data))

        title=data["label"]
        fileList = [f for f in os.listdir("data2/") if os.path.isfile(os.path.join("data2/", f))]
        #print(fileList)
        if any(fileList):
            fileList.sort()
            num = [[int(s) for s in file if s.isdigit()] for file in [fileName for fileName in fileList if title in fileName]]
            num = max([int(''.join(''.join( str(x) for x in numI ))) for numI in num])+1

        with open('data2/'+str(title)+'_'+str(num)+'.json', 'w') as f:
            json.dump(data, f)
            print("saved file to :", 'data/',str(title),'_',str(num),'.json')

    def on_close(self):
        print('connection closed')

    def check_origin(self, origin):
        return True


class WSPred(tornado.websocket.WebSocketHandler):
    def open(self):
        print('new connection')
        print(self.request.remote_ip)
        self.time_last = time.time()
        self.frame_shape = (1, classify.vector_length)
        self.nb_frames = classify.nb_frames
        self.buffer = collections.deque(maxlen=self.nb_frames)
        self.res_buffer = collections.deque(maxlen=int(self.nb_frames/2))
        self.last = ""
        self.send_data = True
        for i in range(self.nb_frames):
            self.buffer.append(np.zeros(self.frame_shape))
        self.counter = 0


    def on_message(self, message):
        #self.counter += 1
        #print(self.counter)
        data=json.loads(message.decode('utf-8'))
        #time_n = time.time()
        #print("time between frames:", time_n-self.time_last)
        #self.time_last = time_n
        Intent = classify.classify(self, data)
        self.res_buffer.append(Intent)
        c = collections.Counter(self.res_buffer)
        sys.stdout.write("\033[K")
        print(Intent, end='\r')
        """if c.most_common()[0][0] is not self.last:
            if c.most_common()[0][1] > self.res_buffer.maxlen/4:
                self.last = c.most_common()[0][0]
                self.send_data = True"""
        if Intent is not "NOP" and Intent is not self.last:
            self.write_message(Intent)
            #self.send_data = False
            self.last = Intent
            #sys.stdout.write("\033[K")
            print("sent", Intent)

    def on_close(self):
        print('connection closed')

    def check_origin(self, origin):
        return True

application = tornado.web.Application([
    (r'/ws', WSHandler),
    (r'/pred', WSPred)
])


if __name__ == "__main__":
    #global time_last
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(8888)
    myIP = socket.gethostbyname('0.0.0.0')
    print('*** Websocket Server Started at %s***' % myIP)
    tornado.ioloop.IOLoop.instance().start()

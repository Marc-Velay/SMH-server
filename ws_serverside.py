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

class WSHandler(tornado.websocket.WebSocketHandler):
    def open(self):
        print('new connection')

    def on_message(self, message):
        num = 0
        data=json.loads(message.decode('utf-8'))
        #print(type(data))

        title=data["label"]
        '''n=0
        mf=Path('data/'+str(n)+'_'+str(title)+'.json')
        while (isfile(mf)):
            n=n+1
            mf=Path('data/'+str(n)+'_'+str(title)+'.json')

        fileList = os.listdir("data/")'''
        #fileList = os.listdir("data/")
        fileList = [f for f in os.listdir("data/") if os.path.isfile(os.path.join("data/", f))]
        print(fileList)
        if any(fileList):
            fileList.sort()
            num = [[int(s) for s in file if s.isdigit()] for file in [fileName for fileName in fileList if title in fileName]]
            num = max([int(''.join(''.join( str(x) for x in numI ))) for numI in num])+1

        with open('data/'+str(title)+'_'+str(num)+'.json', 'w') as f:
            json.dump(data, f)

    def on_close(self):
        print('connection closed')

    def check_origin(self, origin):
        return True

application = tornado.web.Application([
    (r'/ws', WSHandler),
])


if __name__ == "__main__":
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(8888)
    myIP = socket.gethostbyname('0.0.0.0')
    print('*** Websocket Server Started at %s***' % myIP)
    tornado.ioloop.IOLoop.instance().start()

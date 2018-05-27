# -*- coding: utf-8 -*-
__author__ = 'nobita'

# import environment as env
# import accentizer
from flask import Flask, request
import HTMLParser
import tfidf_1_2_balance


# accent = accentizer.accentizer()
# accent.fit(env.DATASET, env.DATATEST)
d = {"NUM":u"NUM: Hỏi về giá tr ịsố", "HUM":u"Hỏi về con người", "LOC":u"Hỏi về địa điểm", "DESC":u"Hỏi về thông tin mô tả", "ABBR":u"Hỏi về chữ viết tắt", "ENTY":u"Hỏi về thực thể"}
d2 = {"abb","exp","animal","body", "color","cremat","currency","dismed","event","food","instru","lang","letter","plant","product","religion","sport","substance","symbol","techmeth","termeq","veh","word","def","manner","reason","gr","ind","title","city","country","mount","state","code","count","date","dist","money","ord","period","perc","speed","temp","volsize","weight","desc","other"}


app = Flask(__name__, static_url_path='',
            static_folder='static',
            template_folder='templates')

@app.route('/', methods = ['GET'])
def homepage():
    return app.send_static_file('index.html')



@app.route('/intent', methods=['POST'])
def process_request():
    data = request.form['data']
    data = HTMLParser.HTMLParser().unescape(data)
    kq = tfidf_1_2_balance.predict_ex(data)
    k1, k2 = kq.split(" ")
    # kq = ngrams.predict_ex(data)
    return "Level 1: " + k1 + "\nLevel 2: " + k2
    # return d[kq]
    # return accent.predict(data)

if __name__ == '__main__':
    app.run('0.0.0.0', port=9100)

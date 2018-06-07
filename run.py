# -*- coding: utf-8 -*-
__author__ = 'nobita'

# import environment as env
# import accentizer
from flask import Flask, request
import HTMLParser
import tfidf_1_2_balance


# accent = accentizer.accentizer()
# accent.fit(env.DATASET, env.DATATEST)
d = {"NUM":u"Hỏi về giá trị số", "HUM":u"Hỏi về con người", "LOC":u"Hỏi về địa điểm", "DESC":u"Hỏi về thông tin mô tả", "ABBR":u"Hỏi về chữ viết tắt", "ENTY":u"Hỏi về thực thể"}
d2 = {"abb":u"Dạng viết tắt","exp":u"ý nghĩa của từ viết tắt","animal":u"động vật","body":u"các bộ phận trên cơ thể", "color":u"màu sắc","cremat":u"phát minh, sách và các sang tạo khác","currency":u"tiền tệ","dismed":u"bệnh tật và y học","event":u"sự kiện","food":u"thức ăn","instru":u"dụng cụ âm nhạc","lang":u"ngôn ngữ","letter":u"kí tự","plant":u"Thực vật","product":u"sản phẩm","religion":u"tôn giáo, tín ngưỡng","sport":u"thể thao","substance":u"nguyên tố, vật chất","symbol":u"biểu tượng và chữ kí","techmeth":u"kỹ thuật và phương pháp","termeq":u"thuật ngữ tương đương","veh":u"phương tiện giao thông","word":u"từ","def":u"định nghĩa","desc":u"mô tả","manner":u"cách thức","reason":u"lý do","gr":u"một nhóm người hoặc một tổ chức","ind":u"cá nhân","title":u"tư cách, danh nghĩa, chức vụ của một người","city":u"thành phố","country":u"đất nước","mount":u"núi","state":u"bang, tỉnh thành","code":u"mã thư tín và các mã khác","count":u"số lượng","date":u"ngày tháng","dist":u"khoảng cách, đo lường tuyến tính","money":u"giá cả","ord":u"thứ hạng","period":u"khoảng thời gian","perc":u"phần trăm","speed":u"tốc độ","temp":u"nhiệt độ","volsize":u"kích thước, diện tích, thể tích","weight":u"cân nặng", "other":u"khác"}
a = ["abb","exp","animal","body", "color","cremat","currency","dismed","event","food","instru","lang","letter","plant","product","religion","sport","substance","symbol","techmeth","termeq","veh","word","def","manner","reason","gr","ind","title","city","country","mount","state","code","count","date","dist","money","ord","period","perc","speed","temp","volsize","weight","desc","other"]
b = [u"Dạng viết tắt", u"ý nghĩa của từ viết tắt", u"động vật", u"các bộ phận trên cơ thể", u"màu sắc"]
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
    return "Level 1: " + k1 + " ("+ d[k1] + ")\nLevel 2: " + k2 + " (" + d2[k2] +")"
    # return d[kq]
    # return accent.predict(data)

if __name__ == '__main__':
    app.run('0.0.0.0', port=5000)

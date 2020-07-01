# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, make_response, jsonify, send_file
import os
import werkzeug
import tempfile

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib

app = Flask(__name__)

# limit upload file size : 32MB
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

UPLOAD_DIR = os.getenv("UPLOAD_DIR_PATH")
UPLOAD_DIR = "./data_space"
print("UPLOAD_DIR={}".format(UPLOAD_DIR))
if UPLOAD_DIR is None:
    UPLOAD_DIR = os.getcwd()
    print("UPLOAD_DIR={}".format(UPLOAD_DIR))



@app.route("/")
def upload():
   import datetime
   now = datetime.datetime.now()
   timeString = now.strftime("%Y-%m-%d %H:%M")
   templateData = {
      'title' : 'HELLO!',
      'name' : 'flask!',
      'ampm' : 1,
      'time': timeString
      }
   #./templates/sample.html
   return render_template('upload.html', **templateData)

@app.route('/data/upload', methods=['POST'])
def upload_multipart():

    if 'uploadFile' not in request.files:
        make_response(jsonify({'result':'uploadFile is required.'}))

    file = request.files['uploadFile']



    print("file={}".format(file))



    #fileName = file.filename
    fileName = 'pred.csv'
    if '' == fileName:
        make_response(jsonify({'result':'filename must not empty.'}))

    saveFileName = werkzeug.utils.secure_filename(fileName)
    print("saveFileName={}".format(saveFileName))
    if (len(saveFileName) == 0):
        fd, tempPath = tempfile.mkstemp()
        print("tempPath={}".format(tempPath))
        saveFileName = os.path.basename(tempPath)
        print("saveFileName={}".format(saveFileName))
        os.close(fd)

    print("UPLOAD_DIR={}".format(UPLOAD_DIR))


    file.save(os.path.join(UPLOAD_DIR, saveFileName))


    return '''
                <html>
        <head>
        <style>
            .a_button {
            display: inline-block;
            border-style: solid;
            background-color: lightgray;
            border-width:1px;
            border-color: darkgray;
            color: black;
            text-decoration: none;
            }
        </style>
        </head>
        
        <body>
        下のボタンを押すと予測データをCSV形式でダウンロード開始！<br>
        <a class="a_button" href="/data/download">ダウンロード</a><br>
        </body></html>


        '''





@app.errorhandler(werkzeug.exceptions.RequestEntityTooLarge)
def handle_over_max_file_size(error):
    print("werkzeug.exceptions.RequestEntityTooLarge")
    return 'result : file size is overed.'

@app.route('/data/download')
def download():



    #ここに予測のコードを入れるか
    df_file = pd.read_csv('./data_space/pred.csv')
    filename = 'finalized_model.sav'
    #test_xの前処理のコード
    df_file.dropna(axis=1, inplace=True)
    df_file = df_file.drop(['動画コメント','（派遣）応募後の流れ','動画タイトル','動画ファイル名','（派遣先）職場の雰囲気','（派遣先）配属先部署'], axis=1)
    le_target_col = ['掲載期間　開始日','休日休暇　備考','期間・時間　勤務時間','勤務地　備考','拠点番号','お仕事名','仕事内容'
                    ,'勤務地　最寄駅1（沿線名）','応募資格','派遣会社のうれしい特典','お仕事のポイント（仕事PR）','勤務地　最寄駅1（駅名）'
                    ,'掲載期間　終了日','期間・時間　勤務開始日']
    le = LabelEncoder()
    for col in le_target_col:
        df_file.loc[:, col] = le.fit_transform(df_file[col])
    #ここまで前処理

    loaded_model = joblib.load(filename)
    result = loaded_model.predict(df_file)
    sub = pd.DataFrame(pd.read_csv("./data_space/pred.csv")['お仕事No.'])

    sub['応募数 合計'] = list(map(np.float64, result))
    sub.to_csv('./data_space/pred.csv', index=False, encoding='utf_8_sig')



    return send_file(os.path.join(UPLOAD_DIR,'pred.csv'),
                     mimetype='text/csv',
                     attachment_filename='pred.csv',
                     as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
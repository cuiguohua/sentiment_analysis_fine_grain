from flask import Flask, request, jsonify
from flask_cors import *
import keras_bert_service
import fasttext_service
import lstm_service
import gcae_service

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.route('/api/sentiment', methods=['POST'])
def sentiment_controller():
    default = [('交通是否便利', '中性'), ('距离商圈远近', '中性'), ('是否容易寻找', '中性'), ('排队等候时间', '中性'), ('服务人员态度', '中性'),
               ('是否容易停车', '中性'),
               ('点菜/上菜速度', '中性'), ('价格水平', '中性'), ('性价比', '中性'), ('折扣力度', '中性'), ('装修情况', '中性'), ('嘈杂情况', '中性'),
               ('就餐空间', '中性'), ('卫生状况', '中性'), ('菜品分量', '中性'), ('口感', '中性'), ('外观', '中性'), ('推荐程度', '中性'),
               ('本次消费感受', '中性'),
               ('再次消费意愿', '中性')]
    print(request.json)
    model = request.json['model']
    content = request.json['content']
    print(model, content)

    if model == 'FastText':
        res = fasttext_service.human_predict(content)
        return jsonify(res), 200
    elif model == 'BERT':
        res = keras_bert_service.human_predict(content)
        return jsonify(res), 200
    elif model == 'LSTM':
        res = lstm_service.human_predict(content)
        return jsonify(res), 200
    elif model == 'GCAE':
        res = gcae_service.human_predict(content)
        return jsonify(res), 200
    else:
        return jsonify(default), 200


if __name__ == '__main__':
    app.run(port=5000, debug=True, host='0.0.0.0')

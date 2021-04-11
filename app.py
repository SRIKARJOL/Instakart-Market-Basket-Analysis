#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import numpy as np
import pickle


# In[2]:


# Loadind required data and model
catboost = pickle.load(open('catboost_model','rb'))
data_test = pickle.load(open('data_test_deploy.pkl','rb'))
product_dict = pickle.load(open('prod_dict.pkl','rb'))


# In[ ]:


# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)

@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    
    order_id_value = request.form["order_id_value"]
    order_number = int(order_id_value)
    data = data_test[data_test.order_id==order_number]
    
    if data.empty:
        # Send empty output
        output = []
    else:    
        predictions = (catboost.predict_proba(data.drop(['product_id','order_id'],axis=1))[:, 1] >= 0.21).astype('int')

        # add the predictions as a new column to the above filtered data
        data['predictions'] = predictions

        # filter out all the products whereever predictions = 1(meaning the product has to be reordered)
        products_id = data[data['predictions'] == 1]['product_id']
    
        prod_id_list = products_id.values.tolist()
    
        output = []
    
        for prod_id in prod_id_list:
            output.append(product_dict[prod_id])
    
    
    return flask.render_template('output.html', predictions=output)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080,debug=False)





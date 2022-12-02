from flask import Flask, redirect, url_for, request
from recommendation_sentiment import recommend_items

app = Flask(__name__)
 
 
@app.route('/success/<name>')
def success(name):
    list_item=recommend_items(name)
    return list_item
 
 
@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        user = request.form['nm']
        return redirect(url_for('success', name=user))
    else:
        user = request.args.get('nm')
        return redirect(url_for('success', name=user))
 
 
if __name__ == '__main__':
    app.run(debug=True)
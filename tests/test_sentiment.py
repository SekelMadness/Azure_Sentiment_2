import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from app import *

negative_text = "It's a bad day, I want to cry because I broke with my girlfriend. It's a poor feeling !"
positive_text = "I am so excited, I love this guy, I have a good feeling !"
def test_negative():
    assert predict_sentiment(negative_text)[0] == "negative"

def test_positive():
    assert predict_sentiment(positive_text)[0] == "positive"
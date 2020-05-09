import json
import requests

def POSTRequester(URL, json_body):
    r = requests.post(URL, json=json_body)
    return r.json()

def GETRequester(URL):
    r = requests.get(URL)
    return r.json()

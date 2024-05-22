import requests

api_key = '8f26525c9c333ad5531e99959ac96f03ca87b46b'
system_id = '55982'

url = f'https://pvoutput.org/service/r2/getoutput.jsp?sid={system_id}&df=20200601&dt=20200630'
headers = {
    'X-Pvoutput-Apikey': api_key,
    'X-Pvoutput-SystemId': system_id
}

response = requests.get(url, headers=headers)
data = response.text

print(data)
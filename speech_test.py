import wave
import requests
import json

def get_token():
    apiKey = "。。。GBOtpg22ZSGAU"
    secretKey = "44。。。e34936227d4a19dc2"

    auth_url = "https://openapi.baidu.com/oauth/2.0/token?grant_type=client_credentials&client_id=" + apiKey + "&client_secret=" + secretKey
    response = requests.get(url=auth_url)
    jsondata = response.text
    return json.loads(jsondata)['access_token']

def use_cloud(token, wavefile):
    fp = wave.open(wavefile, 'rb')
    # 已经录好音的音频片段内容
    nframes = fp.getnframes()
    filelength = nframes*2
    audiodata = fp.readframes(nframes)

    # 百度语音接口的产品ID
    cuid = '71XXXX663'
    server_url = 'http://vop.baidu.com/server_api' + '?cuid={}&token={}'.format(cuid, token)
    headers = {
        'Content-Type': 'audio/pcm; rete=8000',
        'Content-Length': '{}'.format(filelength),
    }

    response = requests.post(url=server_url, headers=headers, data=audiodata)
    return response.text if response.status_code==200 else 'Something Wrong!'



if __name__ == '__main__':
    access_token = get_token()
    print(access_token)
    result = use_cloud(token=access_token, wavefile='./output.wav')
    print(result)


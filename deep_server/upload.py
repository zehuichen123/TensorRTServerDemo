import requests
import time
 
for ii in range(100):
    url = "http://192.168.2.126/api/cls"
    file_path='/home/czh/code/others/cat2.jpg'
    file_name=file_path.split('/')[-1]
    file=open(file_path,'rb')
    files = {'file':(file_name,file,'image/jpg')}
    checkpoint = time.time()
    r = requests.post(url,files=files)
    print("[*] ID %d: Received From GPU server in %g s" % (ii+1, time.time() - checkpoint))
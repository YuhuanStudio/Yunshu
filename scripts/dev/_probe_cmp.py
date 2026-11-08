import json,os,sys
a=json.load(open(sys.argv[1]));b=json.load(open(sys.argv[2]))
for k in sorted(set(a)|set(b)):
    if a.get(k)!=b.get(k): print(k,'|',str(a.get(k))[:110],'|',str(b.get(k))[:110])

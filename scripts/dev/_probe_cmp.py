import json,sys
a=json.load(open(sys.argv[1]));b=json.load(open(sys.argv[2]))
for k in sorted(set(a)&set(b)):
    if a[k]!=b[k]: print(k,'|',str(a[k])[:140],'|',str(b[k])[:140])

from datetime import datetime
from sklearn import metrics 
import matplotlib.pyplot as plt 
import re

import argparse

parser = argparse.ArgumentParser(description='Model generator tool')
parser.add_argument('-l', '--log', help='path to log file')

args = parser.parse_args()

if args.log is None:
   parser.print_help()
   quit()

with open(args.log,'r',errors='replace') as f:
    l=f.readlines()
actual,predicted,Result,instance_number,inf_delay=[],[],0,0,0
for ins in l:
    if len(ins.split(' '))>15:
        inf_delay+=int(ins.split(' ')[13])
    elif len(ins.split(' '))>10:
        inf_delay+=int(ins.split(' ')[10])
    try:
        i=[m.start() for m in re.finditer(':',ins)][-1]
        u=ins.find('-')
        clas=int(ins[u+1:u+2])
        predclas=int(ins[i+2:i+3])
        result=eval(ins[i+4:])
        actual.append(clas)
        predicted.append(predclas)
        if clas==predclas:
            Result+=1
        instance_number+=1
    except:
        pass

print('Number of Instances:',instance_number)
print("Accuracy: {}%".format((Result/len(l))*100))
d1 = datetime.strptime(l[0].split(' ')[1], '%H:%M:%S')
d2 = datetime.strptime(l[instance_number-100].split(' ')[1], '%H:%M:%S')
delay=(d2-d1).total_seconds()
for i in range(100):
    delay+=(int(l[instance_number-99].split(' ')[5])+int(l[instance_number-99].split(' ')[10]))/1000
print("interface delay:",inf_delay/instance_number,'ms')
print("Total delay:",round(delay/60),"Min")
confusion_matrix = metrics.confusion_matrix(actual, predicted) 
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['0','1','2','3','4','5','6','7','8','9']) 
cm_display.plot()
plt.show() 
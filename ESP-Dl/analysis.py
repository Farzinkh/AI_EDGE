from sklearn import metrics 
import matplotlib.pyplot as plt 
import re

with open('REPORT.TXT','r') as f:
    l=f.readlines()
print('Number of Instances:',len(l))
actual,predicted,Result=[],[],0
for ins in l:
    i=[m.start() for m in re.finditer(':',ins)][-1]
    u=ins.find('-')
    clas=int(ins[u+1:u+2])
    predclas=int(ins[i+2:i+3])
    result=eval(ins[i+4:])
    actual.append(clas)
    predicted.append(predclas)
    if result:
        Result+=1

print("Accuracy: {}%".format((Result/len(l))*100))
confusion_matrix = metrics.confusion_matrix(actual, predicted) 
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['0','1','2','3','4','5','6','7','8','9']) 
cm_display.plot()
plt.show() 
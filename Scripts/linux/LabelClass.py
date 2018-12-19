
nLabel, nTrial, nUser, nChannel, nTime  = 4, 40, 32, 40, 8064

print("Program started"+"\n")
fout_labels_class = open("data/label_class_0.dat",'w')

with open('data/labels_0.dat','r') as f:
    for val in f:
        if float(val) > 4.5:
            fout_labels_class.write(str(1) + "\n");
        else:
            fout_labels_class.write(str(0) + "\n");

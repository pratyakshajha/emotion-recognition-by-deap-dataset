import pickle

nLabel, nTrial, nUser, nChannel, nTime  = 4, 40, 32, 40, 8064
def convertData():
    print("Program started"+"\n")
    fout_data = open("data/features_raw.dat",'w')
    fout_labels0 = open("data/labels_0.dat",'w')
    fout_labels1 = open("data/labels_1.dat",'w')
    fout_labels2 = open("data/labels_2.dat",'w')
    fout_labels3 = open("data/labels_3.dat",'w')
    for i in range(8):  #nUser #4, 40, 32, 40, 8064
        if(i%1 == 0):
            if i < 10:
                name = '%0*d' % (2,i+1)
            else:
                name = i+1
        fname = "C:/Users/lumsys/AnacondaProjects/Emo/data/s"+str(name)+".dat"     #C:/Users/lumsys/AnacondaProjects/Emo/
        f = open(fname, 'rb')
        x = pickle.load(f, encoding='latin1')
        print(fname)
    	
        for tr in range(nTrial):
            if(tr%1 == 0):
                for dat in range(nTime):
                    if(dat%32 == 0):
                        for ch in range(nChannel):
                            fout_data.write(str(x['data'][tr][ch][dat]) + " ");
                fout_labels0.write(str(x['labels'][tr][0]) + "\n");
                fout_labels1.write(str(x['labels'][tr][1]) + "\n");
                fout_labels2.write(str(x['labels'][tr][2]) + "\n");
                fout_labels3.write(str(x['labels'][tr][3]) + "\n");
                fout_data.write("\n");
    fout_labels0.close()
    fout_labels1.close()
    fout_labels2.close()
    fout_labels3.close()
    fout_data.close()
    print("\n"+"Print Successful")

if __name__ == '__main__':
    convertData();
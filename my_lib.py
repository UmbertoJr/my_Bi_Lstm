import multiprocessing
import pickle
import numpy as np
import re
import xml.etree.ElementTree as ET
import os

def return_key(row):
        if len(row)==401:
            word = row[0]
        else:
            word = '_'.join(row[:len(row)-400])
        return word
    
def return_vec(row):
        vec = row[len(row)-400:]
        if len(row)==401:
            word = row[0]
        else:
            word = '_'.join(row[:len(row)-400])
        return word, vec
    
def save_obj(obj, name ):
    with open('drive/NLP_project/data/sensembed_vectors/obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

        
def load_obj(name ):
    with open('drive/NLP_project/data/sensembed_vectors/obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    

    
    
def take_vectors(d, word, file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        try:
            f.seek(d[word])
            vec = f.readline().split()
            word, vec = return_vec(vec)
            return word, vec
        except:
            None
            #print('Not "%s" in the dataset'%(word))
    
    
###  build a function that takes as input a word and return a vector
def build_vec(par,senso , row):
    parola = '_'.join([par, senso])
    if parola in row:
        p, vec = take_vectors(row,parola,'drive/NLP_project/data/sensembed_vectors/babelfy_vectors.txt')
    elif par in row:
        p, vec = take_vectors(row,par,'drive/NLP_project/data/sensembed_vectors/babelfy_vectors.txt')
    else:
        if bool(re.findall("_",par)):
            ws = par.split("_")
            vec = np.zeros((400), dtype=float)
            for w in ws:
                if w in row:
                    p, vec_new = take_vectors(row,w,'drive/NLP_project/data/sensembed_vectors/babelfy_vectors.txt')
                    vec = (vec + np.array(vec_new, dtype=float))/2
            return vec.tolist()
        else:
            p, vec = take_vectors(row,"unk",'drive/NLP_project/data/sensembed_vectors/babelfy_vectors.txt')
    return vec

## function that take a word and compute the mean for all sense of that word
def mean_senses(word, row):
    vec = np.zeros((400), dtype=float)
    par = word.attrib["lemma"]
    for sense in word:
        senso = sense.text
        vec = (vec + np.array(build_vec(par, senso, row), dtype=float))/ 2
    return vec


### create a dictionary for each sense.xml file crea le features
def create_dic_for_sense_emb(f, row, modality):
    r = ET.parse("drive/NLP_project/data/"+modality+"/"+f).getroot()
    dic = {}
    for sent in r:
        for word in sent:
            dic[word.tag]= mean_senses(word, row)
    with open("drive/NLP_project/data/"+modality+"/embeddings/"+ re.sub(".xml","",f) + ".pkl", "wb") as file:
        pickle.dump(dic, file, pickle.HIGHEST_PROTOCOL)
        print("done", f)

        
## funzione che crea output per il Deep NN
def build_y(name_file,y_dic,row, modality, domini_visti = {}):
    path = "drive/NLP_project/data/"+modality+"/"
    file = ET.parse(path + name_file)
    r = file.getroot()
    dizionario_per_vettori = {}
    dizionario_per_i_domini = {}
    for sent in r:
        for parola in sent:
            w_to_ds = parola.tag
            for senso in parola:
                if senso.text == y_dic[w_to_ds]:
                    dizionario_per_vettori[w_to_ds] = build_vec(parola.attrib["lemma"], senso.text, row)
                    for el in senso.attrib["Domains"].split(","):
                        dom = re.sub("\{|\}","",el).strip().split("=")
                        dizionario_per_i_domini[w_to_ds] = dom
                        if dom[0] not in domini_visti:
                            domini_visti[dom[0]] = "_"
    dic = [dizionario_per_vettori, dizionario_per_i_domini]
    with open("drive/NLP_project/data/"+modality+"/Y_data/"+ re.sub("file","y_data",re.sub(".xml","",name_file)) + ".pkl", "wb") as file:
        pickle.dump(dic, file, pickle.HIGHEST_PROTOCOL)
        print("done", name_file)        
    return domini_visti     




class create_batch:
    def __init__(self, cosa, row_emb):
        self.path = "drive/NLP_project/data/"
        self.batch = 0
        self.batch_current = 0
        self.done = 0
        self.row_emb = row_emb
        self.mode = cosa
        self.counter = 0
        self.current_file_readed=""
        self.domain = {j: i+1 for i,j in enumerate(sorted(list(pickle.load(open("drive/NLP_project/data/pickle_data.pkl","rb")).keys())))}
        if cosa == "TRAIN":
            self.root = ET.parse(open(self.path + "semcor.data.xml", "r")).getroot()
            self.position_sentence = "d000.s000"
            self.all = sorted([(int(''.join(re.findall("\d", file))),file)for file in os.listdir(self.path + "TRAIN/embeddings")])
        elif cosa == "DEV":
            self.root = ET.parse(open(self.path+"ALL.data.xml", "r")).getroot()
            self.position_sentence = "senseval2.d000.s000"
            self.all = sorted([(int(''.join(re.findall("\d", file))),file)for file in os.listdir(self.path + "DEV/embeddings")])
        
        self.current_dic_input = {}
        self.current_dic_output = []
        self.data = self.take_new_sentences()
        
    def next_sent(self):
        foo = True
        x = y = []
        while len(self.all)>0 and foo:
            try:
                x,y = next(self.data)
                return x,y
                foo = False
            except:
                if len(self.all)==0:
                    foo=False
                try:
                    self.next_batch()
                    self.data = self.take_new_sentences()
                    print("Take new file", self.current_file_readed)    
                except:
                    print("Data is Finished")
                    foo= False    
        
        print("Data is Finished")
        return x,y

    def next_batch(self):
        try:
            next_batch = self.all.pop(0)
        except:
            pass
        self.current_file_readed = next_batch[1]
        self.done += self.batch
        self.batch = next_batch[0] - self.done
        self.batch_current = self.batch
        self.current_dic_input = pickle.load(open(self.path +  self.mode +"/embeddings/"+ next_batch[1], "rb"))
        self.current_dic_output = pickle.load(open(self.path + self.mode +"/Y_data/"+ re.sub("file", "y_data",  next_batch[1]), "rb"))
        
    def take_new_sentences(self):
        counter = 0
        #X = []
        #Y = []
        visto = False
        #non è super efficciente ma funziona
        for doc in self.root:
                for sent in doc:
                    if visto:
                        self.position_sentence = sent.attrib["id"]
                        visto = False
                    if counter>= self.batch_current: #len(X)
                        return None
                        #return((x,y) for x in X for y in Y)
                    
                    
                    if sent.attrib["id"] == self.position_sentence and not visto and counter < self.batch_current:
                        self.counter += 1
                        if len(sent.getchildren())< 50:
                            x,y = self.vectors_extractor(sent)
                            counter +=1
                            yield x,y

                            #X.append(x)
                            #Y.append(y)
                        else:
                            extension_x, extension_y = self.longer_sentence_extractor(sent)
                            self.batch_current += len(extension_x) - 1
                            for x,y in zip(extension_x,extension_y):
                                yield x,y

                        if counter<= self.batch_current:
                            visto = True

                                    
    def vectors_extractor(self, sent):           
        vec_x = []
        vec_y = []
        for word in sent:
            vec_x.append(self.input_data(word))
            vec_y.append(self.output_data(word))
        return (vec_x, vec_y)
    
    def input_data(self, word):
        if word.tag == "instance":
            if word.attrib["id"] in self.current_dic_input:
                return self.current_dic_input[word.attrib["id"]]
        elif word.attrib["lemma"] in self.row_emb:
            return take_vectors(self.row_emb,word.attrib["lemma"],'drive/NLP_project/data/sensembed_vectors/babelfy_vectors.txt')[1]
        else:
            return take_vectors(self.row_emb,"unk",'drive/NLP_project/data/sensembed_vectors/babelfy_vectors.txt')[1]

    def output_data(self, word):
        if word.tag == "instance":
            if len(self.current_dic_output[1][word.attrib["id"]])==2:
                return [self.current_dic_output[0][word.attrib["id"]],self.domain[self.current_dic_output[1][word.attrib["id"]][0]],eval(self.current_dic_output[1][word.attrib["id"]][1]), word.attrib["id"]]
            else:
                return [self.current_dic_output[0][word.attrib["id"]],self.domain[self.current_dic_output[1][word.attrib["id"]][0]],0, word.attrib["id"]]
        elif word.attrib["lemma"] in self.row_emb:
            return [take_vectors(self.row_emb,word.attrib["lemma"],'drive/NLP_project/data/sensembed_vectors/babelfy_vectors.txt')[1], 0, 0, word.attrib["lemma"]]
        else:
            return [take_vectors(self.row_emb,"unk",'drive/NLP_project/data/sensembed_vectors/babelfy_vectors.txt')[1], 0, 0, word.attrib["lemma"]]

    def longer_sentence_extractor(self, sent):
        vec_x = []
        vec_y = []
        words = sent.getchildren()
        numero_di_volte = int(len(words)/50* 3)
        partenza = [int((len(words)-50)/numero_di_volte*i) for i in range(numero_di_volte)]
        for iterat in partenza:
            x,y = self.vectors_extractor(words[iterat:(iterat+50)])
            vec_x.append(x)
            vec_y.append(y)
        return (vec_x , vec_y)
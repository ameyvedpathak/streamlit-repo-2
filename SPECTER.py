import sys
import subprocess

# implement pip as a subprocess:
subprocess.check_call(
    [sys.executable, '-m', 'pip', 'install', 'nltk','sentence-transformers'])
import streamlit as st
import os
import nltk
# from nltk.stem import PorterStemmer
# from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import re
from sklearn.metrics import classification_report,confusion_matrix
#!pip install pyyaml==5.4.1
from tqdm import tqdm
from sentence_transformers import util, SentenceTransformer
import random
import pandas as pd
import plotly.express as px


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')


class BaseClassifier:
    def __init__(self,model_name):

        # self.text_data=data
        # self.labels=labels
        self.model_name=model_name

        self.predictions=[]
        self.preprocessed=[]
        self.labels=[]


        self.lemma=WordNetLemmatizer()
        self.stopwords=stopwords.words('english')

        # self.weights_path=self._make_outdir(folder_name="weights")#create output data folders for each specific model
        # self.results_path=self._make_outdir(folder_name="results")
        self.model_field_name=""


        # print("Pre=processing {} values".format(len(self.text_data)))
        # for w in tqdm(self.text_data):
        #     self.preprocessed.append(self._preprocess(w))


    def set_model_field(self, field_name):

        # print("Set model field name to: {}".format(field_name))
        st.write("Set model field name to:", field_name)
        self.model_field_name=field_name

    def update_data(self, new_preprocessed):
        self.preprocessed=new_preprocessed



    def _make_outdir(self,folder_name="folder"):
        """
        Create output directories
        :param folder_name:
        :return:
        """
        this_path=os.path.join(folder_name,self.model_name)

        if not os.path.exists(this_path):
            os.mkdir(this_path)
            return this_path

    def _get_wordnet_pos(self,word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)

    def preprocess(self, word_list):
        # print("Pre-processing {} values".format(len(word_list)))
        st.write("Pre-processing", len(word_list), " values")
        values=[]
        for w in tqdm(word_list):
             values.append(self._preprocess(w))

        return values



    def _preprocess(self,txt):
        txt=str(txt)
        lemmatised = [self.lemma.lemmatize(w, self._get_wordnet_pos(w)).lower() for w in nltk.word_tokenize(txt)]
        values=[w for w in lemmatised if w not in self.stopwords]
        #print(" ".join(values))
        return " ".join(values)

    def train(self):
        raise NotImplementedError("Please Implement this method")

    def predict(self):
        raise NotImplementedError("Please Implement this method")

    def update_field(self):
        raise NotImplementedError("Please Implement this method")


    def evaluate(self):
        print("Confusion matrix:\n{}".format(confusion_matrix(self.labels, self.predictions)))
        clf=classification_report(self.labels, self.predictions)
        print("Classification Report:\n{}".format(clf))

    def analyse_predictions(self):
        pass




def calculate_similarity_single_sent(pos_idx, cos_sim):
    """
    pos_idx: list with lindex calues of positively-labelled rows
    target_list: list with strings to claculate similarities on

    returns: list of length target_list, with float values between 0 and 1 corresponding to cosine similarity (1=similar).

    Example:
    idx_pos=[0,2,4]
    inputs=["Artificial Intelligence With DEep Learning on COROnary Microvascular Disease",
            "Neuronal Mechanisms of Human Episodic Memory",
            "Polyp REcognition Assisted by a Device Interactive Characterization Tool - The PREDICT Study",
            "Artificial Intelligence With DEep Learning on COROnary Microvascular Disease",
            "Artificial intelligence and machine learning for Covid 19",
            "Effects of a Mindfulness-Based Eating Awareness Training online intervention in adults of obesity: A randomized controlled trials",
            "Prediction of Phakic Intraocular Lens Vault Using Machine Learning",
            "AI system for egg OFC Prediction System of Infants"]

    calculate_similarity_single_sent(idx_pos, inputs,model)

    """
    # emb_source= my_model.encode(target_list)#get our data column

    # print(emb_source.shape)
    # print("Calculating similarity matrix...")

    # print(cos_sim.shape)

    all_sims = []

    for i in tqdm(cos_sim):  # for each input and its pairwise similarities
        avg_for_record = []  # list to store all pairwwise similarities of this field with the positively labelled fields
        for ind in pos_idx:  # for each positive labelled record
            avg_for_record.append(i[ind].item())  # add the cosine similarity
            # print(i[ind])
        all_sims.append(
            sum(avg_for_record) / len(avg_for_record))  # average similarity is used for now, but could use median etc
        # print(sum(avg_for_record)/len(avg_for_record))
        # print("-----")
    # print(all_sims)
    return all_sims

class SPECTER_CLS(BaseClassifier):

    def update_field(self, current_data):
        # print("Loading and making similarity calculations...")
        st.write("Loading and making similarity calculations...")
        self.model_name = 'sentence-transformers/allenai-specter'
        self.model = SentenceTransformer(self.model_name)


        self.emb_source = self.model.encode(list(current_data[self.model_field_name]))  # get our data column
        self.cos_sim = util.pytorch_cos_sim(self.emb_source, self.emb_source)  # .diagonal().tolist()#all similarities
        self.was_prepared = True
        # print(self.cos_sim[0][0])

    def train(self):
        """
        There  is no training for a regular expression filter, but the filter could be reset to something else with each training step
        :param reset_filter:
        :param filter:
        :return:
        """
        pass



    def predict(self, current_data):

        # print("Predicting. Currently discovered {} labels".format(
        #     len(list(current_data[current_data["discovered_labels"] != ""].index.values))))
        st.write("Predicting. Currently discovered {} labels",len(list(current_data[current_data["discovered_labels"] != ""].index.values)))

        # print('Predicting {} data points using <{}> model'.format(len(self.preprocessed),self.model_name))
        self.predictions = []

        idx = list(current_data[current_data[
                                    "discovered_labels"] == 1].index.values)  # filter positive labels and get their index values as list
        if len(idx) >= 10:
            idx_pos = random.choices(idx, k=10)
        else:
            idx_pos = idx
        # print("Found {} labels, predicting based on indexes: {}".format(len(idx),idx_pos))  # print first five as sanity check
        st.write("Found",len(idx),"labels, predicting based on indexes:",idx_pos)
        #print(idx)

        try:
            # sci_title_sim=calculate_similarity_single_sent(idx_pos,self.cos_sim)
            # print(self.was_prepared)
            st.write(self.was_prepared)
            # sci_title_sim=current_data["predictions"]
            # self.emb_source= self.model.encode(list(current_data["ScientificTitle"]))#get our data column
            # self.cos_sim = util.pytorch_cos_sim(self.emb_source, self.emb_source)

            sims = calculate_similarity_single_sent(idx_pos, self.cos_sim)
            sci_title_sim = []
            # print("Index values:")
            st.write("Index values:")
            for data_index in current_data.index.values:
                sci_title_sim.append(sims[data_index])

            #print(sci_title_sim)
        except:
            # print("Resetting an dpreparing data...")
            st.write("Resetting an dpreparing data...")
            self.my_idx = idx_pos
            #current_data.reset_index(drop=True, inplace=True)
            #self.update_field(list(current_data["ScientificTitle"]))  # ScientificTitle#Interventions
            current_data.reset_index(drop=True, inplace=True)
            self.update_field(current_data)
            sci_title_sim = calculate_similarity_single_sent(self.my_idx, self.cos_sim)
            #print(sci_title_sim)
            # print(current_data.index.values[:10])
            st.write(current_data.index.values[:10])

        return sci_title_sim



class ActiveLearner:
    def __init__(self,classifier,data,field="ScientificTitle", time_to_retrain=10,model_name="regex", do_preprocess=True):
        """
        Taking a classifier instance from one of the classifer architectures in classifiers.py, and a dataset
        :param classifier:
        :param data:
        """

        ##############Gold standard (or non-labelled) data
        self.all_data=data
        self.all_data["discovered_labels"]=""#the screened references and discovered gold-standard labels
        self.all_data["predictions"] = 0
        #self.all_data= self.all_data.sample(frac=1, random_state=42)
        self.field=field

        ########################plotting and eval
        self.precisions=[]
        self.recalls=[]
        self.f1s=[]
        self.num_found_list=[]#number of relevant references found at each step
        self.num_steps=0
        self.step_list=[]#number of references screened
        ###################################The classifier with all methods overwritten as needed
        self.do_preprocess=do_preprocess
        self.classifier=classifier

        self.time_to_retrain=time_to_retrain
        self.classifier=classifier(model_name)
        self.change_field()
        # print("Done setting up classifier")
        st.write("Done setting up classifier")

    def change_field(self):
        # print("Setting {} as classification field.".format(self.field))
        st.write("Setting {", self.field, "} as classification field.")

        if self.do_preprocess:
            self.all_data["preprocessed"] = self.classifier.preprocess(self.all_data[self.field])#add back later if we need preprocessing
        else:
            self.all_data["preprocessed"] =self.all_data[self.field]
        self.classifier.set_model_field(self.field)
        #print()

    def retrain(self):
        #print("Retraining and updating")
        self.classifier.update_data(self.all_data["preprocessed"])#update the order of pre-proessed data
        self.classifier.train()

    def reorder(self):
        print("Reordering")
        #print(self.all_data["predictions"])
        self.all_data.sort_values(by=['predictions'], ascending=False, inplace=True)
        #print(list(self.all_data["predictions"])[:20])


    def predict(self):
        #print("Predicting ")
        #print(self.all_data["predictions"].shape)

        self.all_data["predictions"]=self.classifier.predict(self.all_data)
        #print(self.all_data.index.values)


    def discover_labels(self):
        """
        Simulate sreening: Discover true gold-standard labels each time this function is called. Discover the labels for the references on top, which have hopefully been reordered to contain relavant references
        :return:
        """
        df=self.all_data[self.all_data["discovered_labels"] ==""]#not yet screened

        to_discover= list(df.index.values)[:self.time_to_retrain]#get the index values for the items to discover
        for i in to_discover:
            self.all_data.at[i, 'discovered_labels']=self.all_data.at[i, 'label']


    def add_stats(self):
        df = self.all_data[self.all_data["discovered_labels"] == 1]#get positive labelled rows
        #print(df.shape)
        self.num_steps+= self.time_to_retrain#add how many references were screened at this point
        self.step_list.append(self.num_steps)#x axis: that's the variable we will use for plotting! Basically a list of points for x axis
        self.num_found_list.append(df.shape[0])# Y axis: that's the variable we will use for plotting! Basically a list of points for y axis

    def plot_stats(self):
        df=pd.DataFrame(columns=["Screened References", "References found"])
        df["Screened References"]=self.step_list
        df["References found"] = self.num_found_list

        fig = px.line(df, x="Screened References", y="References found", title='Screening progress',template='simple_white')
        chart_data = pd.DataFrame(df, columns=["References found"])
        st.line_chart(chart_data)

        fig.show()



    def reorder_once(self):
        print("Reordering spreadsheet")
        self.time_to_retrain=len(list(self.all_data["label"]))#use all labels that we have

        self.discover_labels()  # "screen" 10 references. In similation, those refs are simply uncovered, ie added to a different column. If we have a screening UI and proper app, this method can be exchanged witha method that asks screeners to provide 10 labels
        self.retrain()  # technically not doing anything right now
        self.predict()  # calculate similarities and predict new order
        self.reorder()  # guess what that does LOL

        return self.all_data

    def simulate_learning(self):
        # print("Simulating active learning")
        st.write("Simulating active learning")
        while "" in list(self.all_data["discovered_labels"]):#while there are sill unlabelled references
            #print(".......Starting iteration...")
            self.discover_labels()#"screen" 10 references. In similation, those refs are simply uncovered, ie added to a different column. If we have a screening UI and proper app, this method can be exchanged witha method that asks screeners to provide 10 labels
            self.retrain()#technically not doing anything right now
            self.predict()#calculate similarities and predict new order
            self.reorder()#guess what that does LOL
            self.add_stats()#adding info to lists for plotting

            #print(self.all_data["discovered_labels"].value_counts())
        self.plot_stats()






# import sys
# import subprocess
#
# # implement pip as a subprocess:
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandas', 'streamlit', 'joblib'])

from ActiveLearner import *
from SPECTER import *
import pandas as pd
import streamlit as st
import pickle
from PIL import Image



@st.cache
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

def main(count=0):

    image = Image.open('IO_logo_new.png')
    st.image(image, width=70, use_column_width='auto')
    st.subheader("AI SCAN")
    menu = ["Filter", "SPECTER", "About"]
    choice = st.sidebar.selectbox("Menu", menu,key=count)
    count +=1

    if choice == "Filter":
        data_file = st.file_uploader("Upload CSV", type=['csv'],key="6")
        global df
        if st.button("Process",key="7"):
            if data_file is not None:
                file_details = {"Filename": data_file.name, "FileType": data_file.type, "FileSize": data_file.size}
                st.write(file_details)
                df = pd.read_csv(data_file)
                st.dataframe(df)

        title = st.text_input('Enter free text for filter or regular expression',"r'(\\bai\\b)|(artificial intelligence)|(machine[\s-]?learn(ing)?)'",key=5) #('Enter free text for filter', 'keywords or regular expression',key="5")
        st.write('Filters used:', title,key="8")

        option = st.selectbox(
            'Select Field',
            ('Scientific Title', 'Intervention', 'Outcomes'),key="9")

        st.write('You selected:', option,key="10")

        # when 'Predict' is clicked, make the prediction and store it
        if st.button("Result",key="11"):

            st.write("Into result section")
            # pickled_model = pickle.load(open('model.pkl', 'rb'))
            # pickled_model.simulate_learning()

            data = pd.read_csv(data_file).fillna("")
            data = data.sample(frac=1, random_state=48)
            data.reset_index(drop=True, inplace=True)
            # st.write(title,key="20")
            classifier = regexClassifier
            # st.write(option,key="20")
            al = ActiveLearner(classifier, data, field="ScientificTitle", model_name="Filter")
            # al = ActiveLearner(classifier, data, field="Interventions", model_name="Filter")
            al.simulate_learning()

            # import streamlit as st
            # import pandas as pd

            #df = pd.read_csv("dir/file.csv")

            output = al.reorder_once()
            csv = convert_df(output)

            st.download_button(
                "Press to Download Results",
                csv,
                "Reordered_file.csv",
                "text/csv",
                key='download-csv'
            )
    elif choice == "SPECTER":
        data_file2 = st.file_uploader("Upload CSV", type=['csv'], key="60")
        global df2
        if st.button("Process", key="70"):
            if data_file2 is not None:
                file_details = {"Filename": data_file2.name, "FileType": data_file2.type, "FileSize": data_file2.size}
                st.write(file_details)
                df2 = pd.read_csv(data_file2)
                st.dataframe(df2)

        option = st.selectbox(
            'Select Field',
            ('Scientific Title', 'Intervention', 'Outcomes'), key="9")

        st.write('You selected:', option, key="10")

        # when 'Predict' is clicked, make the prediction and store it
        if st.button("Result", key="11"):
            st.write("Into result section")
            # pickled_model = pickle.load(open('model.pkl', 'rb'))
            # pickled_model.simulate_learning()

            data = pd.read_csv(data_file2).fillna("")
            data = data.sample(frac=1, random_state=48)
            data.reset_index(drop=True, inplace=True)
            classifier = SPECTER_CLS
            al = ActiveLearner(classifier, data, field="ScientificTitle", model_name="Neural", do_preprocess=True)
            #al = ActiveLearner(classifier, data, field="Interventions", model_name="Neural", do_preprocess=False)###example chosing a different field
            al.simulate_learning()#ecample for simulation, can still be used to provide fancy plot to the user to see how the model would have reacted tto their data in active learning scenario

            output = al.reorder_once()
            csv = convert_df(output)

            st.download_button(
                "Press to Download Results",
                csv,
                "Reordered_file_2.csv",
                "text/csv",
                key='download-csv'
            )


    else:
        st.subheader("About us")
        st.info("*************************************")
        st.write("Under Maintenance")
        # https: // paxful.com /
        # < iframe
        # src = "https://giphy.com/embed/ocuQpTqeFlDOP4fFJI"
        # width = "480"
        # height = "480"
        # frameBorder = "0"
        #
        # class ="giphy-embed" allowFullScreen > < / iframe > < p > < a href="https://giphy.com/gifs/paxful-fixing-under-maintenance-site-ocuQpTqeFlDOP4fFJI" > via GIPHY < / a > < / p >

        st.markdown("![Alt Text](https://media.giphy.com/media/ocuQpTqeFlDOP4fFJI/giphy.gif)")
        st.info("*************************************")

if __name__ == '__main__':
    main()


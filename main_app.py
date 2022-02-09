# import sys
# import subprocess
#
# # implement pip as a subprocess:
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandas', 'streamlit', 'joblib'])

from ActiveLearner import *
import pandas as pd
import streamlit as st
import pickle
from PIL import Image

def main(count=0):
    global df
    image = Image.open('IO_logo.png')
    st.image(image, width=70, use_column_width='auto')
    st.subheader("NIHR Innovation Observatory AI SCAN")
    menu = ["Filter", "SPECTER", "About"]
    choice = st.sidebar.selectbox("Menu", menu,key=count)
    count +=1

    if choice == "Filter":
        st.subheader("Dataset")
        data_file = st.file_uploader("Upload CSV", type=['csv'],key="6")
        if st.button("Process",key="7"):
            if data_file is not None:
                file_details = {"Filename": data_file.name, "FileType": data_file.type, "FileSize": data_file.size}
                st.write(file_details)
                df = pd.read_csv(data_file)
                st.dataframe(df)

        title = st.text_input('Enter free text for filter', 'keywords or regular expression',key="5")
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

            data = df
            data = data.sample(frac=1, random_state=48)
            data.reset_index(drop=True, inplace=True)

            classifier = regexClassifier
            al = ActiveLearner(classifier, data, field="ScientificTitle", model_name="Filter")
            # al = ActiveLearner(classifier, data, field="Interventions", model_name="Filter")
            al.simulate_learning()


    # elif choice == "SPECTER":
    #     st.subheader("Dataset")
    #     data_file = st.file_uploader("Upload CSV", type=['csv'])
    #     if st.button("Process"):
    #         if data_file is not None:
    #             file_details = {"Filename": data_file.name, "FileType": data_file.type, "FileSize": data_file.size}
    #             st.write(file_details)
    #
    #             df = pd.read_csv(data_file)
    #             st.dataframe(df)
    #     option = st.selectbox(
    #         'Select Field',
    #         ('Scientific Title', 'Intervention', 'Outcomes'))
    #
    #     st.write('You selected:', option)

    else:
        st.info("About us")
        st.info("*************************************")


if __name__ == '__main__':
    main()


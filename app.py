import streamlit as st
import pandas as pd 
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from streamlit_option_menu import option_menu
import os
import pycaret
from pycaret import regression, classification



st.set_page_config(page_title="Auto ML", layout="wide")
with st.sidebar:
    selected = option_menu("Menu", ['Home','Upload','---','Analize' ,'Train','---','Model', 'API'], 
        icons=['house', 'cloud-upload',  None, 'bar-chart', 'robot', None, 'download', 'download'], 
        default_index=0)


# if selected =='Home':
#     st.title('Como Utilizar!')


# file=""
print(selected)
if selected == 'Upload':
    st.title('Upload')
    file = st.file_uploader('Only .csv files')

    if file:
        df = pd.read_csv(file)
        st.dataframe(df)


# if selected == 'Analize':
#     st.title('Analise dos seus dados')
#     profile_report = df.profile_report()
#     st_profile_report(profile_report)

# if selected == 'Train':
#     st.title('Treinamento do seu modelo de Machine Learning')
#     target = st.selectbox('Selectione seu Target', df.columns)
#     ignored = st.multiselect('Selectione as variaveis que devem ser ignoradas Target', df.columns)

#     select_reg_class_cluss = st.selectbox('Selectione qual é o tipo de ML', ['Selecione','Classificação','Clusterização','Regressão'])

#     if select_reg_class_cluss == 'Classificacao':

#         classification.setup(df, target=target, ignore_features  =  ignored, silent=True)
#         setup_df = classification.pull()
#         st.dataframe(setup_df)
#         best_model = classification.compare_models()
#         compare_df = classification.pull()
#         st.info('Esse é o modelo de ML')
#         st.dataframe(compare_df)
#         best_model
#         classification.save_model(best_model, 'best_model')

#     if select_reg_class_cluss == 'Clusterização':
#         pass    

#     if select_reg_class_cluss == 'Regressão':
        
#         regression.setup(df, target=target, ignore_features = ignored)
#         setup_df = regression.pull()
#         st.dataframe(setup_df)
#         best_model = regression.compare_models()
#         compare_df = regression.pull()
#         st.info('Esse é o modelo de ML')
#         st.dataframe(compare_df)
#         best_model
#         regression.save_model(best_model, 'best_model')


# if selected == 'Download':
#     with ('best_model.pkl', 'rb') as f:
#         st.download_button('Baixar modelo', f , 'best_model.pkl')


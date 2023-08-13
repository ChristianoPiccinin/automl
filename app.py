import streamlit as st
import pandas as pd 
#import pandas_profiling


from ydata_profiling import ProfileReport
from ydata_profiling.utils.cache import cache_file

from streamlit_pandas_profiling import st_profile_report
import pycaret
from pycaret import classification, regression, clustering
import os



st.set_page_config(page_title="Auto ML", layout="wide")

st.sidebar.title("1. Data")

# Load data
with st.sidebar.expander("Dataset", expanded=True):

    load_options = dict()
    file = st.file_uploader(
            "Upload a csv file", type="csv",
        )

    load_options["separator"] = st.selectbox(
        "What is the separator?", [",", ";", "|"]
    )

    if file:
        df =  pd.read_csv(file, sep=load_options["separator"])
        
    else:
        st.stop()

st.title("Sample Data")
if file:
    st.dataframe(df)

# Cleaning
with st.sidebar.expander("Cleaning", expanded=False,  ):
    
    delete_null = st.checkbox(
        "Delete rows where values is null", disabled= True
    )

    df_original = df.copy(deep=True)
    
    if delete_null:
        df.fillna()
    else:
        df = df_original



st.sidebar.title("2. Analize")    
with st.sidebar.expander("Dashboard", expanded=True):
    active_dash = st.checkbox(
        "Visualize KPI", disabled= False
    )

if active_dash:
    # Generate the Profiling Report
    profile = ProfileReport(
        df, title="Dashboard Dataset", html={"style": {"full_width": True}}, sort=None
    )
    st_profile_report(profile)






        # st.title("Dashboard")
        # profile_report = df.profile_report()
        # st_profile_report(profile_report)
    

st.sidebar.title("3. Modeling")    
with st.sidebar.expander("Dataset", expanded=True):
    pass

st.sidebar.title("4. Setup") 
with st.sidebar.expander("Target", expanded=True):
    target = st.selectbox('Seleect yout Target', df.columns)
    
with st.sidebar.expander("Ignore Features", expanded=True):
    ignored = st.multiselect('Select features to remove', df.columns)

with st.sidebar.expander("Select Problem", expanded=True):
    select_reg_class_cluss = st.selectbox('Select', ['Choose','Regression','Classification','Clustering'])

if select_reg_class_cluss == 'Regression':
    with st.spinner('Wait for it...'):
        st.title("Setup")
        print(df.head())
        regression.setup(df, target=target, ignore_features = ignored)
        setup_df = regression.pull()
        st.dataframe(setup_df)
        best_model = regression.compare_models()
        compare_df = regression.pull()
        st.info('Machine Learning - Regression')
        st.dataframe(compare_df)
        # best_model
        # regression.save_model(best_model, 'best_model')


elif select_reg_class_cluss == 'Classification':
    classification.setup(df, target=target, ignore_features  =  ignored, silent=True)
    setup_df = classification.pull()
    st.dataframe(setup_df)
    best_model = classification.compare_models()
    compare_df = classification.pull()
    print(compare_df)
    st.info('Machine Learning - Classification')
    st.dataframe(compare_df)
    best_model
    classification.save_model(best_model, 'best_model')

elif select_reg_class_cluss == 'Clustering': 
    pass

else:
    st.stop()

    

with st.sidebar.expander("Evaluate", expanded=True):
    active_evaluate= st.checkbox(
        "Evaluate Model", disabled= False
    )


st.sidebar.title("5. Download")    
with st.sidebar.expander("", expanded=True):
    api = st.checkbox(
        "API", disabled= False
    )

    model = st.checkbox(
        "Model", disabled= False
    )



# Column names
# with st.sidebar.expander("Columns", expanded=True):
#     print('')
# # Filtering
# with st.sidebar.expander("Filtering", expanded=False):
#     print('')
# # Resampling
# with st.sidebar.expander("Resampling", expanded=False):
#     print('')
# # Cleaning
# with st.sidebar.expander("Cleaning", expanded=False):
#     print('')
# st.sidebar.title("2. Modelling")

# if selected =='Home':
#     st.title('Como Utilizar!')


# file=""
# print(selected)
# if selected == 'Upload':
#     st.title('Upload')
#     file = st.file_uploader('Only .csv files')

#     if file:
#         df = pd.read_csv(file)
#         st.dataframe(df)


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


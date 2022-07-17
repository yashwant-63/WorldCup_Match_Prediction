"""The main file to predict soccer world cup predictions

The code consists of 3 main parts:
1. Assigning each team data from the csv files to eact team name, and build a dataframe similar to the one the model is trained on.

2. Loading the model and the Preprocess pipeline to process the data including (one hot encoding for categorical variables and Making
PCA for the numerical values for dimensionality reduction

3.Making a web app front end frame work using Streamlit library to deploy the model and put it into production
"""

#Importing the libraries
import pandas as pd
import numpy as np
from numpy import loadtxt
import pickle
import matplotlib.pyplot as plt
import streamlit as st
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score , accuracy_score ,f1_score
from Utilities.func import* 
import os

class DataFrameSelector(BaseEstimator,TransformerMixin):
    def __init__(self,attribute_names):
        self.attribute_names = attribute_names
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        return X[self.attribute_names]

path = os.getcwd()



#loading in the model and the pipeline files to predict on the data for the second model
pickle_in4 = open(path+'/pipeline/model2_xgb.pkl', 'rb')
model2 = pickle.load(pickle_in4)
classes2 = model2.classes_

pickle_in5 = open('pipeline/pipeline2.pkl', 'rb')
pipeline2 = pickle.load(pickle_in5)

pickle_in6 = open('Pipeline/model_goals_xgb.pkl', 'rb')
model3 = pickle.load(pickle_in6)
classes3 = model3.classes_

xgb_preds = loadtxt('Data/xgb_preds.csv', delimiter=',')
ytest = loadtxt('Data/xgb_ytest.csv', delimiter=',')


#create choose list for second model including the teams names from the trained data
team1_list2 = ['Algeria', 'Argentina', 'Australia', 'Belgium', 'Brazil', 'Cameroon',
       'Canada','Chile', 'Colombia', 'Costa Rica', 'Croatia', 'Denmark', 'Ecuador',
       'Egypt', 'England', 'France', 'Germany', 'Ghana', 'Greece', 'Honduras',
       'Iceland', 'Iran', 'Italy', 'Japan','Mexico', 'Morocco', 'Netherlands',
       'New Zealand', 'Nigeria', 'Panama', 'Paraguay', 'Peru', 'Poland',
       'Portugal', 'Qatar','Russia', 'Saudi Arabia', 'Scotland','Senegal', 'Serbia', 'Slovakia',
       'Slovenia', 'South Africa', 'South Korea','Spain', 'Sweden', 'Switzerland', 'Tunisia',
       'Uruguay', 'USA', 'Ukraine', 'United Arab Emirates','Wales']

team2_list2 = team1_list2.copy()


#read the meta data for both home and away teams to assign the data
#based on the choosen team for the second model
df_home = pd.read_csv('Data/df_home_all2.csv',index_col=0)
df_away = pd.read_csv('Data/df_away_all2.csv',index_col=0)
                
def welcome():
	return 'welcome all'

# this is the main function in which we define our webpage
def main():
	# giving the webpage a title
	
	# here we define some of the front end elements of the web page like
	# the font and background color, the padding and the text to be displayed
	html_temp = """
	<div style ="background-color:blue;padding:13px">
	<h2 style ="color:white;text-align:center;">World Cup Match Prediction </h2>
	</div>
	"""

	choices = ['Match Result Prediction','Model Performance']
	ticker = st.sidebar.selectbox('Choose a Page',choices)
	st.markdown(html_temp, unsafe_allow_html = True)

	if (ticker=='Match Result Prediction'):
            # this line allows us to display a drop list to choose team 1 and team 2 
            st.header('Match Prediction Page')
            team_3 = st.selectbox('Team 1', np.array(team1_list2))
            team_4 = st.selectbox('Team 2', np.array(team2_list2))


            # the below line ensures that when the button called 'Predict' is clicked,
            # the prediction function defined above is called to make the prediction
            # and store it in the variable result
            results_df2 = pd.DataFrame()

            # CSS to inject contained in a string
            hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """
            
            if st.button("Predict "):
                if (team_3 == team_4):
                    st.text('Please select different teams')
                else:
                    
                    results_df2 = predict_match_result2(team_3 , team_4,model2,pipeline2, df_home,df_away)

                    # Inject CSS with Markdown
                    st.markdown(hide_table_row_index, unsafe_allow_html=True)

                    #st.dataframe(results_df2)
                    st.table(results_df2.style.format("{:.3f}").hide_index())
                    #this step to preduict the match final result and display the highest results propabilities
                    draw_df , home_w_df , away_w_df = predict_match_result_goals(team_3 , team_4,model3,pipeline2, df_home,df_away)

                    
                    st.subheader('Match Result prediction')

                    
                    #add three dataframes of the match results in case of Draw, Win , Lose
                    col1, col2 ,col3 = st.columns(3)
                    col1.markdown("Draw Results")
                    col1.markdown(hide_table_row_index, unsafe_allow_html=True)
                    col1.table(((pd.DataFrame(draw_df.loc[0].nlargest(3)).T)*(1/(draw_df.loc[0].nlargest(3).values.sum()))).style.format("{:.3f}"))
                    col2.markdown("Team 1 win Results")
                    col2.markdown(hide_table_row_index, unsafe_allow_html=True)
                    col2.table(((pd.DataFrame(home_w_df.loc[0].nlargest(3)).T)*(1/(home_w_df.loc[0].nlargest(3).values.sum()))).style.format("{:.3f}"))
                    col3.markdown("Team 2 win Results")
                    col3.markdown(hide_table_row_index, unsafe_allow_html=True)
                    col3.table(((pd.DataFrame(away_w_df.loc[0].nlargest(3)).T)*(1/(away_w_df.loc[0].nlargest(3).values.sum()))).style.format("{:.3f}"))

                    st.subheader('Teams statistics')

                    col4,col5 = st.columns(2)
                    
                    col4.dataframe(get_team1_stats(team_3,df_home).style.format("{:.2f}"))

                    col5.dataframe(get_team2_stats(team_4,df_away).style.format("{:.2f}"))


	else:
		st.header('Model Performance')
		st.subheader('Performance Metrics')
		score = 'The accuracy score :' + str(np.round(accuracy_score(ytest, xgb_preds),3))
		st.text(score)

		
		score2 = 'The precision score :' + str(np.round(precision_score(ytest, xgb_preds,average='weighted'),3))
		st.text(score2)

		score3 = 'The recall score :' + str(np.round(recall_score(ytest, xgb_preds,average='weighted'),3))
		st.text(score3)

		st.subheader('Confusion Matrix')
		st.pyplot(plot_confusion_matrix(ytest,xgb_preds))

		
	
if __name__=='__main__':
	main()


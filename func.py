import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#functions for model2 data assignment
#Assign values from the dataframe to the team name and retuen a dataframe with all team1 data
def assign_values_to_team3(team,df_home):
    
    if team in df_home.index :
        team1_data =  df_home.loc[team].reset_index()
        team1_data = team1_data.groupby('index').mean().reset_index().rename(columns={'index':'home_team.name'}).iloc[0]
        return team1_data

#Assign values from the dataframe to the team name and retuen a dataframe with all team2 data
def assign_values_to_team4(team,df_away):
    
    if team in df_away.index :
        team2_data =  df_away.loc[team].reset_index()
        team2_data = team2_data.groupby('index').mean().reset_index().rename(columns={'index':'away_team.name'}).iloc[0]
        return team2_data

#run the assign values functions and concat the resultiung 2 dataframes into one dataframe for the model input
def map_inputs_to_data2(team1,team2,df_home,df_away):

    team_3z = assign_values_to_team3(team1,df_home)
               
    team_4z = assign_values_to_team4(team2,df_away)

    input_data = pd.concat([team_3z,team_4z])
    return input_data

#get the input data and preprocess the data using the loaded data processing Pipeline,
#and predict the match result probabilities using predict_proba function, and return a dataframe with the probabilites.
def predict_match_result2(team3 ,team4,model2,pipeline2, df_home,df_away):

    #predict the match result
    input_d = map_inputs_to_data2(team3 , team4,df_home,df_away)
    input_processed = pipeline2.transform(pd.DataFrame(input_d).T)
    preds_test1 = model2.predict_proba(input_processed)

    #predict the match result in case we swaped the teams
    input_d2 = map_inputs_to_data2(team4 , team3,df_home,df_away)
    input_processed2 = pipeline2.transform(pd.DataFrame(input_d2).T)
    preds_test2 = model2.predict_proba(input_processed2)

    #swap the predicted values for team 1 and team 2
    preds_test2[0][2], preds_test2[0][1] = preds_test2[0][1], preds_test2[0][2]

    #calculate the average prediction for both cases 
    preds_test = (preds_test1 + preds_test2)/2
      
    results_df = pd.DataFrame(columns=model2.classes_,data=np.round(preds_test,3))
    results_df.rename(columns={0:'Draw Probability',1:'{} wins Probability'.format(team3),2:'{} wins Probability'.format(team4)},inplace=True)
    return results_df

#Predict function for the final match result prediction
def predict_match_result_goals(team3 ,team4,model3,pipeline2,df_home,df_away):

    #predict the match score probability
    input_d = map_inputs_to_data2(team3 , team4,df_home,df_away)
    input_processed = pipeline2.transform(pd.DataFrame(input_d).T)
    preds_test1 = model3.predict_proba(input_processed)

    #predict the match score probability in case we swaped the teams
    input_d2 = map_inputs_to_data2(team4 , team3,df_home,df_away)
    input_processed2 = pipeline2.transform(pd.DataFrame(input_d2).T)
    preds_test2 = model3.predict_proba(input_processed2)

    #swap the predicted values for team 1 and team 2
    preds_test2[0][2], preds_test2[0][1] = preds_test2[0][1], preds_test2[0][2]

    #calculate the average prediction for both cases 
    preds_test = (preds_test1 + preds_test2)/2
      
    results_df = pd.DataFrame(columns=model3.classes_,data=np.round(preds_test,4))
    draw_df = results_df[[x for x in results_df.columns if (int(x[0]) == int(x[2]))]]
    home_w_df = results_df[[x for x in results_df.columns if (int(x[0]) > int(x[2]))]]
    away_w_df = results_df[[x for x in results_df.columns if (int(x[0]) < int(x[2]))]]

    return draw_df,home_w_df,away_w_df

#function to display confusion matrix
def plot_confusion_matrix(y_test,preds):
    fig, ax = plt.subplots(figsize=(6, 6))
    conf_matrix = confusion_matrix(y_test,preds)
    
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='large')
     
    plt.xlabel('Predictions', fontsize=15)
    plt.ylabel('Actuals', fontsize=15)
    ticks = ['Draw','Team1 Win','Team2 Win']
    labels= [0,1,2]
    plt.xticks(labels,ticks)
    plt.yticks(labels,ticks)
    plt.title('Confusion Matrix', fontsize=16)
    
    return fig

def get_team1_stats(team,df_home):
    s = assign_values_to_team3(team,df_home)
    s.index = s.index.str.replace('home_team.', 'Team 1 ')
    s.index = s.index.str.replace('home_', '')
    s.rename(index={'90s':'Avg. Minutes played' , 'Gls':'Avg. Goals','Ast':'Avg. Assists','CrdY':'Avg. Yellow Cards','CrdR':'Avg. Red Cards','Team 1 Total value':'Total Value in Million €'},inplace=True)
    s = pd.DataFrame(s)
    s.columns = [s.iloc[0][0]]
    s.loc['Total Value in Million €'] = s.loc['Total Value in Million €']/1000000
    s.loc['Avg. Minutes played'] = s.loc['Avg. Minutes played'] * 90
    s = s.loc[['Team 1 Rank','Team 1  age','Total Value in Million €','Team 1 Points','Avg. Minutes played','Avg. Goals','Avg. Assists',
              'Avg. Yellow Cards','Avg. Red Cards']]
    return s

def get_team2_stats(team,df_away):
    s = assign_values_to_team4(team,df_away)
    s.index = s.index.str.replace('away_team.', 'Team 2 ')
    s.index = s.index.str.replace('away_', '')
    s.rename(index={'90s':'Avg. Minutes played' , 'Gls':'Avg. Goals','Ast':'Avg. Assists','CrdY':'Avg. Yellow Cards','CrdR':'Avg. Red Cards','Team 2 Total value':'Total Value in Million €'},inplace=True)
    s = pd.DataFrame(s)
    s.columns = [s.iloc[0][0]]
    s.loc['Total Value in Million €'] = s.loc['Total Value in Million €']/1000000
    s.loc['Avg. Minutes played'] = s.loc['Avg. Minutes played'] * 90
    s = s.loc[['Team 2 Rank','Team 2  age','Total Value in Million €','Team 2 Points','Avg. Minutes played','Avg. Goals','Avg. Assists',
              'Avg. Yellow Cards','Avg. Red Cards']]
    return s

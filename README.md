# Data mining project for ISYE-7406

# 2022 Soccer World Cup Prediction App

## Data Sources:
• **Teams General statistics** from https://fbref.com, which includes teams statistics for :
  1. World cup 2018 (https://fbref.com/en/comps/1/stats/FIFA-World-Cup-Stats)
  2. World cup 2014 (https://fbref.com/en/comps/1/1709/stats/2014-FIFA-World-Cup-Stats)
  3. world cup 2010 (https://fbref.com/en/comps/1/19/stats/2010-FIFA-World-Cup-Stats)
  4. Friendly macthes for years 2020-2021-2022 
    (https://fbref.com/en/comps/218/3697/stats/2020-Friendlies-M-Stats)
    (https://fbref.com/en/comps/218/10979/stats/2021-Friendlies-M-Stats)
    (https://fbref.com/en/comps/218/stats/Friendlies-M-Stats)

#### The main prediction features are :


  - G-PK: non penalty goals
  - PI : number of players
  - MP : matches played
  - 90s : minutes played divided by 90
  - GLs : goals
  - Ast : assistance
  - PK :penalty kicks made
  - PKatt : penalty kicks attempted
  - CrdY : yellow cards
  - CrdR : red cards
  - G+A : goals + assists
  - G-PK.1 :goals minus penalty kicks
  - G+A-PK : goals + assists minus penalty kicks

For duplicate team names we used the average statistics of different datasets.

• **Matches Results from 2010 to 2018 from kaggle**
  (https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
  and i filtered the data to contain only data from 2010
  
• **Match results from 2017 to 2022 from API** (data in the API are only from 2017): 
  (http://livescore-api.com/)
  
  We collected the historical data for world cup 2018, friendly matches, all continents competitions as (UEFA,Copa America, Asian Cup, etc...)
  
 • **Fifa Rank data** (from https://www.transfermarkt.com/statistik/weltrangliste)
  The data contains:
  1. Team rank
  2. Squad size
  3. Total value (in Euros)
  4. Average Age
  5. Points

The matches results data from 2010 to 2018 was merged with the new data from API to get all matches data from 2010 to 2022, 
and added the rank data and average teams statistics to each team in one dataframe

## Libraries Used:
1. Pandas (for data processing)
2. Matplotlib (for visualizations for checking the data)
3. Scikit Learn (for data preprocessing and data split into train and test data)
4. Xgboost (to build the classification models)
5. json (to load API data)
6. urllib.request (to open the URL and receive the API response)
7. pickle (to save the trained models)

## Data Processing:
1. Teams' names are checked to be the same in different datasets.
2. datasets are merged together to form one dataset contains matches results from 2010 and average statitics for each team.
3. For the teams average stats, if there are many records of each team we calculate the average. and each team stats are merged to the team results data.
4. New Column created to show the match result as 1 if Team 1 wins, 2 if Team2 wins and 0 for draw.
5. We made 2 special dataframes for Team1 and Team2 statistics so when the user inputs the team name we map the statistics data from these dataframes to be ready for the model input.
6. We use Preprocess pipeline to OneHotEncode the categorical variables as Team names, and create Principla component analysis for dinensionality reduction.

## Model building
Two models are built.
1. The first model to predict the match winner and provide the probability of each team winning.
We used 80% of the data for training and 20% for testing. We tried Logistic Regression and XGBoost Classifier 
with Cross validation and Hyperparameters tuning for better results.
We decided to go with the XGboost model as it gives little higher accuracy (57%).

2.The second model is to predict the match score and gives the highest 3 probabilities of score in each case Win, Lose, Draw.

## Web app with Streamlit
The web app has 2 pages
1. The first page has 2 dropdown menus for teams selection, and predict the match winner probability and match score probability and display the results in tables, and the average statistics of each team with :
- Team Rank (Fifa rank updated on march 2022)
- Team average Age (Fifa updated on march 2022)
- Team Total value in Euros (Fifa updated on march 2022)
- Team points (Rank is calculated by Fifa updated on march 2022)
- Average matches played in each Tournament.
- Average goals in each Tournament.
- Average assists in each Tournament.
- Average goals in each Tournament.
- Average yellow cards in each Tournament.
- Average red cards in each Tournament.

2. The second page displays the first model evaluation metrics as Accuracy, precision, Recall ,and confusion matrix.


## Requirements.txt 
the requirement file has all libraries used and their versions, to install the libraries for the deployment process.

## Dockerlfile 
includes the run command to run the file with streamlit






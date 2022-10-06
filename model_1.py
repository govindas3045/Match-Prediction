import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, skellam
import statsmodels.api as smf
import statsmodels.formula.api as sm

#inputs
home_team = 'Man City'
away_team = 'Norwich'


#setting up data
def setup_data():
	df_path = 'C:/Python/Python 3/Match Prediction/data.csv' #path to data

	df = pd.read_csv(df_path)
	df = df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]
	df = df.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})
	# print(df)
	Global_HomeGoals_mean = df.mean()['HomeGoals']
	Global_AwayGoals_mean = df.mean()['AwayGoals']


	#plotting a Poisson dist of 'Home team goals' and 'Away team goals' using the calculated means as lambda values
	# x = []
	# for i in range(0,9):
	# 	x.append(i)
	# x = np.array(x)

	# plt.bar(x, poisson.pmf(x, Global_HomeGoals_mean),label="Home")
	# plt.title("Poisson - Home team")
	# plt.xlabel("Goals per match")
	# plt.ylabel("Frequency")
	# plt.show()

	# plt.bar(x, poisson.pmf(x, Global_AwayGoals_mean))
	# plt.title("Poisson - Away team")
	# plt.xlabel("Goals per match")
	# plt.ylabel("Frequency")
	# plt.show()
	# plt.show()


	#adding Goal Difference column to data
	df['GD'] = df['HomeGoals'] - df['AwayGoals']
	# print(df['GD'].unique())

	#plotting Goal Difference distribution of data
	# y = []
	# for i in range(-6,6):
	# 	y.append(int(i))
	# y = np.array(y)
	# plt.hist(df[['GD']].values, range(-6,6), alpha=0.7,density = True)
	# plt.show()


	#creating final set of data
	goal_model_data = pd.concat([df[['HomeTeam','AwayTeam','HomeGoals']].assign(home=1).rename(
	            columns={'HomeTeam':'team', 'AwayTeam':'opponent','HomeGoals':'goals'}),
	           df[['AwayTeam','HomeTeam','AwayGoals']].assign(home=0).rename(
	            columns={'AwayTeam':'team', 'HomeTeam':'opponent','AwayGoals':'goals'})]) #now we have two datasets in 1, home and away teams swapped
	# print(goal_model_data.head(), goal_model_data.shape)
	# print(goal_model_data[370:])

	return goal_model_data

data = setup_data()

#fitting a glm model using the data
poisson_model = sm.glm(formula="goals ~ home + team + opponent", data=data, family=smf.families.Poisson()).fit()
# print(poisson_model.summary())

#calculating win, draw and lose porbabilities
def game(HT, AT):
	exp_goals_HT = poisson_model.predict(pd.DataFrame(data={'team':HT,'opponent':AT,'home':1}, index=[1]))
	exp_goals_AT = poisson_model.predict(pd.DataFrame(data={'team':AT,'opponent':HT,'home':0}, index=[1]))

	prob_home_win = float(1-skellam.cdf(0,exp_goals_HT,exp_goals_AT))
	prob_away_win = float(skellam.cdf(-1,exp_goals_HT,exp_goals_AT))
	prob_draw = float(1-prob_home_win-prob_away_win)

	print()
	print(f'{HT} Win: {round(prob_home_win,4)*100}%')
	print(f'Draw: {round(prob_draw,4)*100}%')
	print(f'{AT} Win: {round(prob_away_win,4)*100}%\n')
	print(f'{HT} Expected Goals: {exp_goals_HT.values}')
	print(f'{AT} Expected Goals: {exp_goals_AT.values}')

game(home_team, away_team)
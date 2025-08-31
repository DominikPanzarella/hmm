import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from hmmlearn.hmm import GMMHMM
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

spy = pd.read_csv("/Users/dominikpanzarella/PycharmProjects/hmm/resources/SPY.csv")
gld = pd.read_csv("/Users/dominikpanzarella/PycharmProjects/hmm/resources/GLD.csv")
spy_tr = pd.read_csv("/Users/dominikpanzarella/PycharmProjects/hmm/resources/SPY_TR.csv")

for df in [spy, gld, spy_tr]:
    df.columns = [col.capitalize() for col in df.columns]
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

spy_weekly = spy['Close'].resample('W-FRI').last()
spy_tr_weekly = spy_tr['Close'].resample('W-FRI').last()
gld_weekly = gld['Close'].resample('W-FRI').last()

spy_returns = np.log(spy_weekly / spy_weekly.shift(1)).dropna().to_frame(name='SPY_Return')
spy_tr_returns = np.log(spy_tr_weekly / spy_tr_weekly.shift(1)).to_frame(name='SPY_TR_Return')
gld_returns = np.log(gld_weekly / gld_weekly.shift(1)).dropna().to_frame(name='GLD_Return')
combined = spy_returns.join(gld_returns, how='inner')
combined = combined.join(spy_tr_returns, how='inner')

combined['Volatility'] = combined['SPY_Return'].rolling(window=4).std()
combined.dropna(inplace=True)

# == STRATEGY PARAMETER ==
window_size = 104       #weeks
n_components = 2        #number of states
n_min = 3               #number of guassian

exception_count = 0
strategy_returns = []

# === WALK-FORWARD LOOP ===
for i in range(window_size, len(combined) - 1):
    train_data = combined.iloc[i - window_size:i]
    test_index = combined.index[i]

    X_train = train_data[['SPY_Return', 'Volatility']].values
    scaler = StandardScaler()
    #train_data = scaler.fit_transform(train_data_raw[['SPY_Return']])

    X_train_scaled = scaler.fit_transform(X_train)

    try:
        # model = GaussianHMM(n_components=n_components, covariance_type='diag', n_iter=100, random_state=0)
        model = GMMHMM(n_components=n_components,
                       n_mix=n_min,
                       covariance_type='tied',
                       n_iter=100,
                       random_state=0)
        model.fit(X_train_scaled)

        # Use scaled data directly
        posteriors = model.predict_proba(X_train_scaled)
        pi_t = posteriors[-1]
        pi_next = pi_t @ model.transmat_

        # Predict states
        states = model.predict(X_train)

        train_data = train_data.copy()
        train_data['State'] = states
        state_returns = train_data.groupby('State')['SPY_Return'].mean()
        high_risk_state = state_returns.idxmin()

        # Allocation
        weight_gld = pi_next[high_risk_state]
        weight_spy = 1 - weight_gld
    except:
        weight_gld = 0.5
        weight_spy = 0.5
        exception_count += 1



    r_spy = combined.loc[test_index, 'SPY_Return']
    r_gld = combined.loc[test_index, 'GLD_Return']

    #strategy_return = weight_spy * r_spy + weight_gld * r_gld
    strategy_return = np.log(weight_spy * np.exp(r_spy) + weight_gld * np.exp(r_gld))
    strategy_returns.append((test_index, strategy_return, r_spy, r_gld, weight_gld))

# === RESULTS ===

strategy_df = pd.DataFrame(strategy_returns, columns=['Date', 'Strategy_Return', 'SPY_Return', 'GLD_Return', 'Weight_GLD'])
strategy_df.set_index('Date', inplace=True)

strategy_df['Cumulative_Strategy'] = (1 + strategy_df['Strategy_Return']).cumprod()
strategy_df['Cumulative_SPY'] = (1 + strategy_df['SPY_Return']).cumprod()
strategy_df['Cumulative_GLD'] = (1 + strategy_df['GLD_Return']).cumprod()
strategy_df['Mixed_Return'] = np.log(0.5 * np.exp(strategy_df['SPY_Return']) + 0.5 * np.exp(strategy_df['GLD_Return']))

strategy_df['Cumulative_Mixed'] = (1 + strategy_df['Mixed_Return']).cumprod()

# === PLOT ====
strategy_df[['Cumulative_Strategy', 'Cumulative_SPY', 'Cumulative_GLD', 'Cumulative_Mixed']].plot(figsize=(12, 8))
plt.title("Walk-Forward Gaussian GMMHMM Strategy: Probabilistic Risk Allocation")
plt.ylabel("Cumulative Return")
plt.grid(True)
plt.tight_layout()
plt.show()

# === STATS ===
def compute_stats(returns, freq='W'):
        periods_per_year = {'D': 252, 'W': 52, 'M':12}[freq]

        cumulative_return = (1 + returns).prod()
        n_periods = len(returns)
        years = n_periods / periods_per_year
        cagr = cumulative_return ** (1 / years) - 1

        rolling = (1 + returns).cumprod()
        peak = rolling.cummax()
        drawdown = (rolling - peak) / peak
        max_dd = drawdown.min()
        avg_dd = drawdown[drawdown < 0].mean()

        volatility = returns.std() * np.sqrt(periods_per_year)
        sharpe = returns.mean() / returns.std() * np.sqrt(periods_per_year)

        return {
            'CAGR': cagr,
            'Max Drawdown': max_dd,
            'Avg Drawdown': avg_dd,
            'Sharpe ratio': sharpe,
        }

stats_strategy = compute_stats(strategy_df['Strategy_Return'], freq='W')
stats_spy = compute_stats(strategy_df['SPY_Return'], freq='W')
stats_gld = compute_stats(strategy_df['GLD_Return'], freq='W')
stats_mxd= compute_stats(strategy_df['Mixed_Return'], freq='W')

report = pd.DataFrame([stats_strategy, stats_spy, stats_gld, stats_mxd], index=['Strategy_Return', 'SPY_Return', 'GLD_Return', 'Mixed_Return'])

print(report)
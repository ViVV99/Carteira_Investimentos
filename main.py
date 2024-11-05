import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

# Configuração de visualização
sns.set(style='whitegrid')

# Parâmetros da carteira
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # Substitua pelos tickers desejados
start_date = '2020-01-01'
end_date = '2023-01-01'
risk_tolerance = 0.5  # Tolerância de risco (desvio padrão máximo permitido)

# 1. Coleta dos dados
def get_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    returns = data.pct_change().dropna()
    return returns

returns = get_data(tickers, start_date, end_date)

# 2. Funções de Otimização
# Função para calcular o retorno e risco (volatilidade)
def portfolio_performance(weights, returns):
    portfolio_return = np.dot(weights, returns.mean()) * 252  # Retorno anualizado
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))  # Volatilidade anualizada
    return portfolio_return, portfolio_volatility

# Função de minimização para o risco (objetivo de maximizar retorno/risco)
def minimize_volatility(weights, returns):
    return portfolio_performance(weights, returns)[1]

# Função para realizar a otimização
def optimize_portfolio(returns, risk_tolerance):
    num_assets = len(returns.columns)
    args = (returns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Soma dos pesos = 1
    bounds = tuple((0, 1) for _ in range(num_assets))  # Pesos entre 0 e 1
    init_guess = num_assets * [1. / num_assets]  # Chute inicial igual para todos os ativos

    # Minimização da volatilidade para o risco desejado
    optimized = minimize(minimize_volatility, init_guess, args=args,
                         method='SLSQP', bounds=bounds, constraints=constraints)

    # Verificação da tolerância de risco
    weights = optimized.x
    _, portfolio_volatility = portfolio_performance(weights, returns)
    
    if portfolio_volatility > risk_tolerance:
        print("Não foi possível otimizar a carteira dentro do limite de risco.")
        return None
    
    return weights

# Obtendo os pesos otimizados
weights = optimize_portfolio(returns, risk_tolerance)
if weights is not None:
    print("Pesos Otimizados da Carteira:", weights)

# 3. Visualização dos Resultados
def plot_results(weights, returns):
    # Fronteira eficiente
    portfolio_returns, portfolio_volatility = portfolio_performance(weights, returns)
    plt.figure(figsize=(10, 6))
    plt.scatter(portfolio_volatility, portfolio_returns, c='red', label="Carteira Otimizada")
    plt.xlabel('Volatilidade (Risco)')
    plt.ylabel('Retorno Esperado')
    plt.title('Fronteira Eficiente')
    plt.legend()

    # Gráfico de Pizza para Distribuição dos Ativos
    plt.figure(figsize=(8, 8))
    plt.pie(weights, labels=returns.columns, autopct='%1.1f%%', startangle=140)
    plt.title('Distribuição dos Ativos na Carteira')

    # Evolução dos Retornos Acumulados
    cumulative_returns = (1 + returns @ weights).cumprod()
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns, label="Carteira Otimizada")
    plt.xlabel('Data')
    plt.ylabel('Retorno Acumulado')
    plt.title('Evolução do Retorno da Carteira')
    plt.legend()
    plt.show()

# Plotagem dos gráficos
if weights is not None:
    plot_results(weights, returns)

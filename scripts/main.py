from src.classes.environment import TradingEnvironment

if __name__ == "__main__":
    trading_environment = TradingEnvironment()
    while trading_environment.k < trading_environment.N:
        trading_environment.step()
    trading_environment.plot_simulation()
    print("Hello World!")

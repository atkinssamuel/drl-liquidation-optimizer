from src.classes.environment import TradingEnvironment
from src.classes.constant import ConstantStrategy


if __name__ == "__main__":
    trading_environment = TradingEnvironment()
    strategy = ConstantStrategy()
    while trading_environment.k < trading_environment.N:
        n = strategy.get_share_quantity(trading_environment)
        trading_environment.step(n)
    trading_environment.plot_simulation()
    print("Hello World!")

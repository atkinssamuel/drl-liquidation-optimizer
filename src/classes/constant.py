class ConstantStrategy:
    @staticmethod
    def get_share_quantity(trading_environment):
        return trading_environment.X / trading_environment.N

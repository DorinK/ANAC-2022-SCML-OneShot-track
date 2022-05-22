#!/usr/bin/env python
from collections import defaultdict

from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

from .other_agents.agent_team86 import AgentOneOneTwo
from .other_agents.agent_template import LearningAgent


QUANTITY = 0
TIME = 1
UNIT_PRICE = 2

from negmas import Outcome, ResponseType

__all__ = [
    "LinearRegressionAgent",
    "LearningAverageAgent",
    "RollingAverageAgent",
]


class LinearRegressionAgent(AgentOneOneTwo):

    def _price_range(self, nmi):
        mn, mx = super()._price_range(nmi)
        if self._is_selling(nmi):
            if len(self._best_selling) > 0:
                x = np.array([i for i in range(len(self._best_selling))]).reshape(-1, 1)
                reg = LinearRegression().fit(x, np.array(self._best_selling).reshape(-1, 1))
                mn = max(mn, reg.predict(np.array([len(x) + 1]).reshape(-1, 1)).item())
        else:
            if len(self._best_buying) > 0:
                x = np.array([i for i in range(len(self._best_buying))]).reshape(-1, 1)
                reg = LinearRegression().fit(x, np.array(self._best_buying).reshape(-1, 1))
                mx = min(mx, reg.predict(np.array([len(x)+1]).reshape(-1, 1)).item())

        return mn, mx


class LearningAverageAgent(LearningAgent):
    def init(self):
        """Initialize the quantities and best prices received so far"""
        super().init()
        self._all_acc_selling, self._all_acc_buying = [], []
        self._all_opp_selling = defaultdict(list)
        self._all_opp_buying = defaultdict(lambda: [])
        self._all_opp_acc_selling = defaultdict(list)
        self._all_opp_acc_buying = defaultdict(lambda: [])

    def before_step(self):
        super().before_step()
        self._all_selling, self._all_buying = [], []

    def step(self):
        """Initialize the quantities and best prices received for next step"""
        super().step()
        self._all_opp_selling = defaultdict(list)
        self._all_opp_buying = defaultdict(lambda: [])

    def on_negotiation_success(self, contract, mechanism):
        """Record sales/supplies secured"""
        super().on_negotiation_success(contract, mechanism)

        # update my current best price to use for limiting concession in other
        # negotiations
        up = contract.agreement["unit_price"]
        if self._is_selling(mechanism):
            partner = contract.annotation["buyer"]
            self._all_acc_selling.append(up)
            self._all_opp_acc_selling[partner].append(up)
        else:
            partner = contract.annotation["seller"]
            self._all_acc_buying.append(up)
            self._all_opp_acc_buying[partner].append(up)

    def respond(self, negotiator_id, state, offer):
        # find the quantity I still need and end negotiation if I need nothing more
        response = super().respond(negotiator_id, state, offer)
        # update my current best price to use for limiting concession in other
        # negotiations
        ami = self.get_nmi(negotiator_id)
        up = offer[UNIT_PRICE]
        if self._is_selling(ami):
            partner = ami.annotation["buyer"]
            self._all_opp_selling[partner].append(up)
            self._all_selling.append(offer[UNIT_PRICE])
        else:
            partner = ami.annotation["seller"]
            self._all_opp_buying[partner].append(up)
            self._all_buying.append(offer[UNIT_PRICE])
        return response

    def _price_range(self, ami):
        mn = ami.issues[UNIT_PRICE].min_value
        mx = ami.issues[UNIT_PRICE].max_value

        if self._is_selling(ami):
            partner = ami.annotation["buyer"]
            self._best_selling = mean(self._all_selling, mx)
            self._best_acc_selling = mean(self._all_acc_selling, mx)
            self._best_opp_selling[partner] = mean(self._all_opp_selling[partner], mx)
            self._best_opp_acc_selling[partner] = mean(self._all_opp_acc_selling[partner], mx)
        else:
            partner = ami.annotation["seller"]
            self._best_buying = mean(self._all_buying, mn)
            self._best_acc_buying = mean(self._all_acc_buying, mn)
            self._best_opp_buying[partner] = mean(self._all_opp_buying[partner], mn)
            self._best_opp_acc_buying[partner] = mean(self._all_opp_acc_buying[partner], mn)
        return super()._price_range(ami)


class RollingAverageAgent(AgentOneOneTwo):
    def _price_range(self, nmi):
        mn, mx = super()._price_range(nmi)
        if self._is_selling(nmi):
            mn = max(mn, mean(self._best_selling, mn))
        else:
            mx = min(mx, mean(self._best_buying, mx))
        return mn, mx


def mean(list_items, default):
    if len(list_items) == 0:
        return default
    else:
        rolling_last_value = pd.Series(list_items).ewm(alpha=0.1, adjust=False).mean().iloc[-1]
        return default if pd.isna(rolling_last_value) else rolling_last_value


if __name__ == "__main__":
    from other_agents.agent_template import try_agent, print_type_scores, LearningAgent

    world, ascores, tscores = try_agent(RollingAverageAgent)
    print_type_scores(tscores)


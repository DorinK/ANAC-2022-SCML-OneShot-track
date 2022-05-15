#!/usr/bin/env python
import statistics
from sklearn.linear_model import LinearRegression
import numpy as np
from scml.oneshot import OneShotAgent

from other_agents.agent_team86 import AgentOneOneTwo

QUANTITY = 0
TIME = 1
UNIT_PRICE = 2

from negmas import Outcome, ResponseType

__all__ = [
    "RegressionAgent"
]

class RegressionAgent(AgentOneOneTwo):

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

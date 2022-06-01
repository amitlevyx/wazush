import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import pycountry

from IMLearn.base import BaseEstimator
from powerpuff.agoda_cancellation_estimator import AgodaCancellationEstimator
from powerpuff.utils.cancellation_code import evaluate_cancellation_code, no_show, fine_after_x_days, \
    is_full_price_at_order, cancellation_complexity
from powerpuff.utils.currencies import convert_currency
from scipy.stats import stats

